// Package router selects the SGLang node for a request.
//
// The original router hashed the system-prompt fingerprint modulo the pool size and
// never looked at node health. This version keeps the same idea (same system prompt
// -> same node, so SGLang's RadixAttention cache stays warm) and makes it production
// safe:
//
//  1. Candidates are nodes in the target group that pass the role filter, are up
//     (active health checks), whose circuit breaker admits traffic, and that this
//     request has not already tried.
//  2. Candidates are ranked by rendezvous (highest-random-weight) hashing, so losing
//     one node only moves that node's keys instead of remapping almost everything.
//  3. The first-ranked node under its bounded-load limit wins; a hot node spills to
//     the next rank rather than queueing (a cache miss is cheaper than a saturated GPU).
//  4. If every candidate is at capacity the request waits in a bounded queue.
package router

import (
	"cmp"
	"container/list"
	"context"
	"errors"
	"hash/fnv"
	"log"
	"math"
	"slices"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/unicoregpu/radixgates/gateway/breaker"
	"github.com/unicoregpu/radixgates/gateway/config"
)

var (
	ErrNoNode       = errors.New("no SGLang instance available")
	ErrQueueFull    = errors.New("admission queue full")
	ErrQueueTimeout = errors.New("timed out waiting for node capacity")

	errAtCapacity = errors.New("all available nodes at capacity")
	// errPreferredBusy means the top-ranked node is at its concurrency limit and the request should
	// wait for it (see Options.AffinityWait) instead of spilling to another node.
	errPreferredBusy = errors.New("preferred node at capacity")
)

// Node represents one SGLang inference server.
type Node struct {
	Group         string
	Role          string // "" = combined | "prefill" = prefill-only | "decode" = decode-only
	URL           string
	Name          string // host:port, used in logs and metric labels
	MaxConcurrent int

	nameHash uint64
	inflight int // guarded by Router.mu
	placed   int // prefixes currently placed on this node, guarded by Router.mu
	up       atomic.Bool
	br       *breaker.Breaker // nil when the breaker is disabled
}

func (n *Node) Up() bool                    { return n.up.Load() }
func (n *Node) SetUp(v bool) (changed bool) { return n.up.Swap(v) != v }

func (n *Node) BreakerState() breaker.State {
	if n.br == nil {
		return breaker.Closed
	}
	return n.br.State()
}

func (n *Node) available() bool { return n.up.Load() && (n.br == nil || n.br.Available()) }

func (n *Node) allow() (breaker.Ticket, bool) {
	if n.br == nil {
		return breaker.Ticket{}, true
	}
	return n.br.Allow()
}

// Options configures routing and admission.
type Options struct {
	BoundedLoadFactor float64
	AffinityFloor     int
	AffinityWait      time.Duration // how long to wait for a full preferred node before spilling; 0 spills at once
	PlacementSize     int           // prompt prefixes remembered per (group, prefix); 0 disables placement memory
	MaxQueue          int
	QueueTimeout      time.Duration
	Breaker           *breaker.Config                           // nil disables circuit breaking
	OnBreakerChange   func(node string, from, to breaker.State) // optional, must not block
}

// Router holds the pool of SGLang nodes.
type Router struct {
	opts  Options
	nodes []*Node

	mu      sync.Mutex
	notify  chan struct{} // closed and replaced on every release (broadcast)
	waiting int

	placeMap map[uint64]*list.Element // placement key -> element of placeLRU, guarded by mu
	placeLRU *list.List               // *placement, most recently used first
}

type placement struct {
	key  uint64
	node *Node
}

// New builds a Router from the gateway config.
func New(instances []config.SGLangInstance, defaultMaxConcurrent int, opts Options) *Router {
	if opts.BoundedLoadFactor < 1 {
		opts.BoundedLoadFactor = 1
	}
	if opts.AffinityFloor < 1 {
		opts.AffinityFloor = 1
	}
	nodes := make([]*Node, 0, len(instances))
	for _, inst := range instances {
		mc := inst.MaxConcurrent
		if mc <= 0 {
			mc = defaultMaxConcurrent
		}
		name := hostPort(inst.Host, inst.Port)
		n := &Node{Group: inst.Group, Role: inst.Role, URL: "http://" + name, Name: name, MaxConcurrent: mc, nameHash: hashString(name)}
		n.up.Store(true)
		if opts.Breaker != nil {
			cb := opts.OnBreakerChange
			n.br = breaker.New(*opts.Breaker, breaker.WithOnChange(func(from, to breaker.State) {
				if cb != nil {
					cb(name, from, to)
				}
			}))
		}
		log.Printf("[Router] Node registered: group=%q role=%q url=%s max_concurrent=%d", n.Group, n.Role, n.URL, mc)
		nodes = append(nodes, n)
	}
	r := &Router{opts: opts, nodes: nodes, notify: make(chan struct{})}
	if opts.PlacementSize > 0 {
		r.placeMap = make(map[uint64]*list.Element)
		r.placeLRU = list.New()
	}
	return r
}

func hostPort(host string, port int) string { return host + ":" + strconv.Itoa(port) }

func (r *Router) Nodes() []*Node { return r.nodes }

// EntryPoint is the default node filter: prefill-only nodes are not request entry points.
func EntryPoint(n *Node) bool { return n.Role != "prefill" }

// Lease is a granted slot on a node. Call exactly one of Done or Release.
type Lease struct {
	N    *Node
	t    breaker.Ticket
	r    *Router
	once sync.Once
}

// Done frees the slot and reports the node-side outcome to the breaker.
func (l *Lease) Done(success bool) {
	l.once.Do(func() {
		if l.N.br != nil {
			l.N.br.Done(l.t, success)
		}
		l.r.release(l.N)
	})
}

// Release frees the slot without a verdict (client cancelled, node said "busy").
func (l *Lease) Release() {
	l.once.Do(func() {
		if l.N.br != nil {
			l.N.br.Cancel(l.t)
		}
		l.r.release(l.N)
	})
}

func (r *Router) release(n *Node) {
	r.mu.Lock()
	n.inflight--
	close(r.notify)
	r.notify = make(chan struct{})
	r.mu.Unlock()
}

// Acquire picks a node for prefixHash and waits in the bounded queue when every
// candidate is at capacity. filter may be nil (defaults to EntryPoint); exclude
// lists node names this request already tried.
func (r *Router) Acquire(ctx context.Context, prefixHash, targetGroup string, filter func(*Node) bool,
	exclude map[string]struct{}) (*Lease, error) {
	if filter == nil {
		filter = EntryPoint
	}
	kh := hashString(prefixHash)
	queued := false
	var deadline, affinity <-chan time.Time
	hold := r.opts.AffinityWait > 0 // keep waiting for the preferred node until the affinity wait ends

	r.mu.Lock()
	for {
		l, err := r.tryPickLocked(kh, targetGroup, filter, exclude, hold)
		if err == nil {
			if queued {
				r.waiting--
			}
			r.mu.Unlock()
			return l, nil
		}
		if err != errAtCapacity && err != errPreferredBusy {
			if queued {
				r.waiting--
			}
			r.mu.Unlock()
			return nil, err
		}
		if !queued {
			if r.waiting >= r.opts.MaxQueue {
				r.mu.Unlock()
				return nil, ErrQueueFull
			}
			r.waiting++
			queued = true
			t := time.NewTimer(r.opts.QueueTimeout)
			defer t.Stop()
			deadline = t.C
		}
		if err == errPreferredBusy && affinity == nil {
			t := time.NewTimer(r.opts.AffinityWait)
			defer t.Stop()
			affinity = t.C
		}
		ch := r.notify
		r.mu.Unlock()

		select {
		case <-ch:
		case <-affinity:
			hold, affinity = false, nil
		case <-deadline:
			r.mu.Lock()
			r.waiting--
			r.mu.Unlock()
			return nil, ErrQueueTimeout
		case <-ctx.Done():
			r.mu.Lock()
			r.waiting--
			r.mu.Unlock()
			return nil, ctx.Err()
		}
		r.mu.Lock()
	}
}

type scored struct {
	n *Node
	s uint64
}

func (r *Router) tryLeaseLocked(n *Node) (*Lease, bool) {
	t, ok := n.allow()
	if !ok {
		return nil, false
	}
	n.inflight++
	return &Lease{N: n, t: t, r: r}, true
}

func (r *Router) tryPickLocked(kh uint64, group string, filter func(*Node) bool, exclude map[string]struct{},
	holdForPreferred bool) (*Lease, error) {
	cands := make([]scored, 0, len(r.nodes))
	total := 0
	for _, n := range r.nodes {
		if group != "" && n.Group != group {
			continue
		}
		if !filter(n) {
			continue
		}
		if _, ex := exclude[n.Name]; ex || !n.available() {
			continue
		}
		cands = append(cands, scored{n, score(kh, n.nameHash)})
		total += n.inflight
	}
	if len(cands) == 0 {
		return nil, ErrNoNode
	}
	slices.SortFunc(cands, func(x, y scored) int { return cmp.Compare(y.s, x.s) })
	if r.opts.PlacementSize > 0 {
		// The placed node (or, for a new prefix, the node holding the fewest prefixes) becomes the
		// preferred node; the rest keep their rendezvous order as the spill order.
		if i := r.placeLocked(mix64(kh^hashString(group)), cands, exclude); i > 0 {
			moved := cands[i]
			copy(cands[1:i+1], cands[:i])
			cands[0] = moved
		}
	}

	limit := int(math.Ceil(r.opts.BoundedLoadFactor * float64(total+1) / float64(len(cands))))
	limit = max(limit, r.opts.AffinityFloor)

	denied := false
	for i, c := range cands {
		if c.n.inflight < min(c.n.MaxConcurrent, limit) {
			if l, ok := r.tryLeaseLocked(c.n); ok {
				return l, nil
			}
			denied = true
		} else if i == 0 && holdForPreferred && c.n.inflight >= c.n.MaxConcurrent && c.n.inflight < limit {
			// The preferred node is full but not over the hot-spot bound: wait for a slot on it.
			return nil, errPreferredBusy
		}
	}
	// Preferred nodes are over their bounded-load limit: spill to the least loaded.
	var best *Node
	for _, c := range cands {
		if c.n.inflight < c.n.MaxConcurrent && (best == nil || c.n.inflight < best.inflight) {
			best = c.n
		}
	}
	if best != nil {
		if l, ok := r.tryLeaseLocked(best); ok {
			return l, nil
		}
		denied = true
	}
	if denied {
		return nil, ErrNoNode // breaker refused between Available() and Allow()
	}
	return nil, errAtCapacity
}

// placeLocked returns the index in cands of the node this placement key should go to and records the
// placement. A key stays where it is placed while that node is a candidate. If its node is only excluded
// because this request already failed there, another node serves the retry and the placement is kept; if
// its node is unavailable the key moves to the node holding the fewest keys. Caller holds r.mu.
func (r *Router) placeLocked(key uint64, cands []scored, exclude map[string]struct{}) int {
	if el, ok := r.placeMap[key]; ok {
		e := el.Value.(*placement)
		for i, c := range cands {
			if c.n == e.node {
				r.placeLRU.MoveToFront(el)
				return i
			}
		}
		if _, excluded := exclude[e.node.Name]; excluded && e.node.available() {
			return leastPlaced(cands)
		}
	}
	best := leastPlaced(cands)
	r.setPlacementLocked(key, cands[best].n)
	return best
}

// leastPlaced picks the candidate holding the fewest placed keys, then the fewest in flight; ties keep
// the better rendezvous rank because cands is sorted.
func leastPlaced(cands []scored) int {
	best := 0
	for i := 1; i < len(cands); i++ {
		a, b := cands[i].n, cands[best].n
		if a.placed < b.placed || (a.placed == b.placed && a.inflight < b.inflight) {
			best = i
		}
	}
	return best
}

func (r *Router) setPlacementLocked(key uint64, n *Node) {
	if el, ok := r.placeMap[key]; ok {
		e := el.Value.(*placement)
		if e.node != n {
			e.node.placed--
			n.placed++
			e.node = n
		}
		r.placeLRU.MoveToFront(el)
		return
	}
	r.placeMap[key] = r.placeLRU.PushFront(&placement{key: key, node: n})
	n.placed++
	for r.placeLRU.Len() > r.opts.PlacementSize {
		last := r.placeLRU.Back()
		e := last.Value.(*placement)
		e.node.placed--
		delete(r.placeMap, e.key)
		r.placeLRU.Remove(last)
	}
}

// Ranking returns node names ordered by rendezvous score for key (health ignored).
func (r *Router) Ranking(key string) []string {
	kh := hashString(key)
	sc := make([]scored, len(r.nodes))
	for i, n := range r.nodes {
		sc[i] = scored{n, score(kh, n.nameHash)}
	}
	slices.SortFunc(sc, func(x, y scored) int { return cmp.Compare(y.s, x.s) })
	out := make([]string, len(sc))
	for i, c := range sc {
		out[i] = c.n.Name
	}
	return out
}

// Status is a point-in-time view of one node.
type Status struct {
	Name     string `json:"name"`
	Group    string `json:"group"`
	Role     string `json:"role"`
	Up       bool   `json:"up"`
	Breaker  string `json:"breaker"`
	Inflight int    `json:"inflight"`
	Max      int    `json:"max_concurrent"`
	Placed   int    `json:"placed_prefixes"`
}

func (r *Router) Snapshot() []Status {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]Status, len(r.nodes))
	for i, n := range r.nodes {
		out[i] = Status{n.Name, n.Group, n.Role, n.Up(), n.BreakerState().String(), n.inflight, n.MaxConcurrent, n.placed}
	}
	return out
}

func (r *Router) QueueDepth() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.waiting
}

// AnyAvailable reports whether at least one entry-point node could take a request now.
func (r *Router) AnyAvailable() bool {
	for _, n := range r.nodes {
		if EntryPoint(n) && n.available() {
			return true
		}
	}
	return false
}

func hashString(s string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(s))
	return mix64(h.Sum64())
}

// mix64 is the splitmix64 finalizer; it decorrelates the FNV output.
func mix64(x uint64) uint64 {
	x ^= x >> 30
	x *= 0xbf58476d1ce4e5b9
	x ^= x >> 27
	x *= 0x94d049bb133111eb
	x ^= x >> 31
	return x
}

func score(keyHash, nameHash uint64) uint64 { return mix64(keyHash ^ nameHash) }
