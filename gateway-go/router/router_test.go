package router

import (
	"context"
	"errors"
	"fmt"
	"hash/fnv"
	"testing"
	"time"

	"github.com/unicoregpu/radixgates/gateway/breaker"
	"github.com/unicoregpu/radixgates/gateway/config"
)

func instances(n int, group string) []config.SGLangInstance {
	out := make([]config.SGLangInstance, n)
	for i := range out {
		out[i] = config.SGLangInstance{Group: group, Host: "127.0.0.1", Port: 30000 + i}
	}
	return out
}

// origIndex is the mapping the original router used (see the upstream-snapshot tag).
func origIndex(key string, n int) int {
	h := fnv.New32a()
	_, _ = h.Write([]byte(key))
	return int(h.Sum32()) % n
}

func opts() Options {
	return Options{BoundedLoadFactor: 1.25, AffinityFloor: 4, MaxQueue: 8, QueueTimeout: 200 * time.Millisecond}
}

func newRouter(n, maxConc int, o Options) *Router { return New(instances(n, "local"), maxConc, o) }

func acquire(t *testing.T, r *Router, key string, ex map[string]struct{}) *Lease {
	t.Helper()
	l, err := r.Acquire(context.Background(), key, "", nil, ex)
	if err != nil {
		t.Fatalf("Acquire(%q): %v", key, err)
	}
	return l
}

func TestAffinityIsStable(t *testing.T) {
	r := newRouter(4, 8, opts())
	for k := 0; k < 50; k++ {
		key := fmt.Sprintf("system-prompt-%d", k)
		first := acquire(t, r, key, nil)
		want := first.N.Name
		first.Done(true)
		for i := 0; i < 5; i++ {
			l := acquire(t, r, key, nil)
			if l.N.Name != want {
				t.Fatalf("key %q moved from %s to %s at low load", key, want, l.N.Name)
			}
			l.Done(true)
		}
	}
}

// Losing 1 of 4 nodes must not move keys owned by the surviving nodes (the original
// hash % N router moved about 75% of them).
func TestSurvivingKeysDoNotMove(t *testing.T) {
	four := newRouter(4, 8, opts())
	three := newRouter(3, 8, opts()) // ports 30000..30002: node 30003 removed
	moved, survivors, modMoved := 0, 0, 0
	for k := 0; k < 20000; k++ {
		key := fmt.Sprintf("prefix-%d", k)
		before := four.Ranking(key)[0]
		if before == "127.0.0.1:30003" {
			continue
		}
		survivors++
		if three.Ranking(key)[0] != before {
			moved++
		}
		if origIndex(key, 4) != origIndex(key, 3) { // the original router: FNV-32a(key) % pool size
			modMoved++
		}
	}
	t.Logf("surviving keys remapped after losing 1 of 4: rendezvous=%.3f hash%%N=%.3f",
		float64(moved)/float64(survivors), float64(modMoved)/float64(survivors))
	if moved != 0 {
		t.Fatalf("rendezvous moved %d surviving keys", moved)
	}
	if float64(modMoved)/float64(survivors) < 0.5 {
		t.Fatalf("baseline unexpectedly stable")
	}
}

func TestSkipsDownAndOpenNodes(t *testing.T) {
	o := opts()
	o.Breaker = &breaker.Config{FailureThreshold: 1, OpenDuration: time.Hour, HalfOpenMax: 1}
	r := newRouter(4, 8, o)
	key := "some-prefix"
	pref := r.Ranking(key)
	byName := map[string]*Node{}
	for _, n := range r.Nodes() {
		byName[n.Name] = n
	}
	byName[pref[0]].SetUp(false)
	l := acquire(t, r, key, nil)
	if l.N.Name != pref[1] {
		t.Fatalf("with %s down want %s, got %s", pref[0], pref[1], l.N.Name)
	}
	l.Done(false) // breaker (threshold 1) opens on pref[1]
	l2 := acquire(t, r, key, nil)
	if l2.N.Name != pref[2] {
		t.Fatalf("want %s, got %s", pref[2], l2.N.Name)
	}
	l2.Done(true)
	byName[pref[0]].SetUp(true)
	l3 := acquire(t, r, key, nil)
	if l3.N.Name != pref[0] {
		t.Fatalf("affinity must return once healthy, got %s", l3.N.Name)
	}
	l3.Done(true)
}

func TestGroupAndRoleFiltersAreKept(t *testing.T) {
	inst := []config.SGLangInstance{
		{Group: "local", Host: "127.0.0.1", Port: 1, Role: "prefill"},
		{Group: "local", Host: "127.0.0.1", Port: 2},
		{Group: "remote", Host: "127.0.0.1", Port: 3},
	}
	r := New(inst, 4, opts())
	for k := 0; k < 30; k++ {
		l, err := r.Acquire(context.Background(), fmt.Sprintf("k%d", k), "", nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		if l.N.Role == "prefill" {
			t.Fatal("a prefill-only node must never be a request entry point")
		}
		l.Done(true)
	}
	l, err := r.Acquire(context.Background(), "x", "remote", nil, nil)
	if err != nil || l.N.Group != "remote" {
		t.Fatalf("group filter ignored: %v %+v", err, l)
	}
	l.Done(true)
	if _, err := r.Acquire(context.Background(), "x", "nope", nil, nil); !errors.Is(err, ErrNoNode) {
		t.Fatalf("unknown group: want ErrNoNode, got %v", err)
	}
}

func TestExcludeSet(t *testing.T) {
	r := newRouter(3, 8, opts())
	pref := r.Ranking("x")
	l := acquire(t, r, "x", map[string]struct{}{pref[0]: {}})
	if l.N.Name != pref[1] {
		t.Fatalf("want %s, got %s", pref[1], l.N.Name)
	}
	l.Done(true)
	all := map[string]struct{}{}
	for _, n := range r.Nodes() {
		all[n.Name] = struct{}{}
	}
	if _, err := r.Acquire(context.Background(), "x", "", nil, all); !errors.Is(err, ErrNoNode) {
		t.Fatalf("want ErrNoNode, got %v", err)
	}
}

func TestBoundedLoadSpills(t *testing.T) {
	o := opts()
	o.AffinityFloor = 2
	r := newRouter(4, 8, o)
	pref := r.Ranking("hot")
	var leases []*Lease
	for i := 0; i < 2; i++ {
		l := acquire(t, r, "hot", nil)
		if l.N.Name != pref[0] {
			t.Fatalf("request %d spilled early to %s", i, l.N.Name)
		}
		leases = append(leases, l)
	}
	l := acquire(t, r, "hot", nil)
	if l.N.Name == pref[0] {
		t.Fatal("third concurrent request should spill off the saturated preferred node")
	}
	leases = append(leases, l)
	for _, l := range leases {
		l.Done(true)
	}
}

func TestQueueWaitsFullTimesOutAndCancels(t *testing.T) {
	o := opts()
	o.MaxQueue = 1
	o.QueueTimeout = 80 * time.Millisecond
	r := newRouter(1, 1, o)
	held := acquire(t, r, "k", nil)

	errCh := make(chan error, 1)
	go func() {
		_, err := r.Acquire(context.Background(), "k", "", nil, nil)
		errCh <- err
	}()
	for r.QueueDepth() != 1 {
		time.Sleep(time.Millisecond)
	}
	if _, err := r.Acquire(context.Background(), "k", "", nil, nil); !errors.Is(err, ErrQueueFull) {
		t.Fatalf("want ErrQueueFull, got %v", err)
	}
	if err := <-errCh; !errors.Is(err, ErrQueueTimeout) {
		t.Fatalf("want ErrQueueTimeout, got %v", err)
	}
	if r.QueueDepth() != 0 {
		t.Fatalf("queue depth leaked: %d", r.QueueDepth())
	}

	// a waiter is woken by a release
	o.QueueTimeout = 2 * time.Second
	r2 := newRouter(1, 1, o)
	h2 := acquire(t, r2, "k", nil)
	got := make(chan error, 1)
	go func() {
		l, err := r2.Acquire(context.Background(), "k", "", nil, nil)
		if err == nil {
			l.Done(true)
		}
		got <- err
	}()
	for r2.QueueDepth() != 1 {
		time.Sleep(time.Millisecond)
	}
	h2.Done(true)
	select {
	case err := <-got:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("waiter not woken by release")
	}

	// context cancel while queued
	ctx, cancel := context.WithCancel(context.Background())
	r3 := newRouter(1, 1, o)
	h3 := acquire(t, r3, "k", nil)
	defer h3.Done(true)
	cerr := make(chan error, 1)
	go func() {
		_, err := r3.Acquire(ctx, "k", "", nil, nil)
		cerr <- err
	}()
	for r3.QueueDepth() != 1 {
		time.Sleep(time.Millisecond)
	}
	cancel()
	if err := <-cerr; !errors.Is(err, context.Canceled) {
		t.Fatalf("want context.Canceled, got %v", err)
	}
	if r3.QueueDepth() != 0 {
		t.Fatal("queue depth leaked after cancel")
	}
	held.Done(true)
}

func TestReleaseTwiceIsSafe(t *testing.T) {
	r := newRouter(1, 2, opts())
	l := acquire(t, r, "k", nil)
	l.Done(true)
	l.Release()
	l.Done(false)
	if s := r.Snapshot()[0]; s.Inflight != 0 {
		t.Fatalf("inflight = %d", s.Inflight)
	}
}

// A node's prefix cache only helps if the requests for that prefix keep arriving at the node. When the
// preferred node is merely full, a request may wait for it for AffinityWait instead of spilling.
func TestAffinityWaitHoldsForThePreferredNode(t *testing.T) {
	o := opts()
	o.AffinityWait = 500 * time.Millisecond
	o.QueueTimeout = 2 * time.Second
	r := newRouter(2, 1, o)
	pref := r.Ranking("k")[0]
	first := acquire(t, r, "k", nil)
	if first.N.Name != pref {
		t.Fatalf("first lease on %s, want the preferred node %s", first.N.Name, pref)
	}
	got := make(chan string, 1)
	go func() {
		l, err := r.Acquire(context.Background(), "k", "", nil, nil)
		if err != nil {
			got <- "error: " + err.Error()
			return
		}
		got <- l.N.Name
		l.Release()
	}()
	time.Sleep(60 * time.Millisecond) // the second request is now waiting for the preferred node
	first.Release()
	select {
	case n := <-got:
		if n != pref {
			t.Fatalf("waiting request went to %s, want the preferred node %s", n, pref)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("waiting request never got a node")
	}
}

func TestAffinityWaitExpiresAndSpills(t *testing.T) {
	o := opts()
	o.AffinityWait = 80 * time.Millisecond
	o.QueueTimeout = 2 * time.Second
	r := newRouter(2, 1, o)
	pref := r.Ranking("k")[0]
	first := acquire(t, r, "k", nil)
	defer first.Release()
	start := time.Now()
	l := acquire(t, r, "k", nil)
	defer l.Release()
	if l.N.Name == pref {
		t.Fatal("second request should have spilled to the other node")
	}
	if d := time.Since(start); d < 60*time.Millisecond || d > time.Second {
		t.Fatalf("spilled after %v, want about the affinity wait (80ms)", d)
	}
}

func TestWithoutAffinityWaitAFullPreferredNodeSpillsAtOnce(t *testing.T) {
	r := newRouter(2, 1, opts())
	pref := r.Ranking("k")[0]
	first := acquire(t, r, "k", nil)
	defer first.Release()
	start := time.Now()
	l := acquire(t, r, "k", nil)
	defer l.Release()
	if l.N.Name == pref || time.Since(start) > 40*time.Millisecond {
		t.Fatalf("expected an immediate spill to the other node, got %s after %v", l.N.Name, time.Since(start))
	}
}

func TestAffinityWaitDoesNotHoldForADownNode(t *testing.T) {
	o := opts()
	o.AffinityWait = 5 * time.Second
	r := newRouter(2, 1, o)
	pref := r.Ranking("k")[0]
	for _, n := range r.Nodes() {
		if n.Name == pref {
			n.SetUp(false)
		}
	}
	start := time.Now()
	l := acquire(t, r, "k", nil)
	defer l.Release()
	if l.N.Name == pref || time.Since(start) > 100*time.Millisecond {
		t.Fatalf("a down preferred node must not be waited for: got %s after %v", l.N.Name, time.Since(start))
	}
}

func TestAffinityWaitStillObeysTheQueueTimeout(t *testing.T) {
	o := opts()
	o.AffinityWait = 5 * time.Second
	o.QueueTimeout = 120 * time.Millisecond
	r := newRouter(1, 1, o)
	first := acquire(t, r, "k", nil)
	defer first.Release()
	start := time.Now()
	_, err := r.Acquire(context.Background(), "k", "", nil, nil)
	if !errors.Is(err, ErrQueueTimeout) {
		t.Fatalf("got %v, want ErrQueueTimeout", err)
	}
	if d := time.Since(start); d > time.Second {
		t.Fatalf("queue timeout took %v", d)
	}
	if r.QueueDepth() != 0 {
		t.Fatalf("queue depth %d after the timeout, want 0", r.QueueDepth())
	}
}

// The hot-spot bound still wins: a preferred node that is over its bounded-load limit is not waited for.
func TestAffinityWaitDoesNotHoldAHotKeyPastTheLoadBound(t *testing.T) {
	o := opts()
	o.BoundedLoadFactor = 1
	o.AffinityFloor = 1
	o.AffinityWait = 5 * time.Second
	r := newRouter(2, 1, o)
	pref := r.Ranking("k")[0]
	first := acquire(t, r, "k", nil)
	defer first.Release()
	start := time.Now()
	l := acquire(t, r, "k", nil)
	defer l.Release()
	if l.N.Name == pref || time.Since(start) > 100*time.Millisecond {
		t.Fatalf("a key over the load bound must spill at once: got %s after %v", l.N.Name, time.Since(start))
	}
}

// Placement memory: a small set of hot prefixes should be spread evenly over the nodes and then stay put,
// instead of following the hash (which can put most of them on one node).
func placementOpts() Options {
	o := opts()
	o.PlacementSize = 64
	return o
}

func nodeByName(r *Router, name string) *Node {
	for _, n := range r.Nodes() {
		if n.Name == name {
			return n
		}
	}
	return nil
}

func placed(r *Router, name string) int {
	for _, s := range r.Snapshot() {
		if s.Name == name {
			return s.Placed
		}
	}
	return -1
}

func TestPlacementSpreadsNewPrefixesEvenlyAndKeepsThemThere(t *testing.T) {
	o := placementOpts()
	o.PlacementSize = 64
	r := newRouter(3, 4, o)
	first := map[string]string{}
	for i := 0; i < 30; i++ { // plain hashing would spread 30 keys unevenly over 3 nodes
		key := fmt.Sprintf("prompt-%d", i)
		l := acquire(t, r, key, nil)
		first[key] = l.N.Name
		l.Release()
	}
	for _, s := range r.Snapshot() {
		if s.Placed != 10 {
			t.Fatalf("node %s holds %d prefixes, want 10 each: %+v", s.Name, s.Placed, r.Snapshot())
		}
	}
	for round := 0; round < 5; round++ {
		for key, node := range first {
			l := acquire(t, r, key, nil)
			if l.N.Name != node {
				t.Fatalf("%s moved from %s to %s", key, node, l.N.Name)
			}
			l.Release()
		}
	}
}

func TestPlacementMovesAKeyWhenItsNodeIsDownAndKeepsItThereAfterRecovery(t *testing.T) {
	r := newRouter(3, 4, placementOpts())
	l := acquire(t, r, "k", nil)
	x := l.N.Name
	l.Release()
	nodeByName(r, x).SetUp(false)
	l = acquire(t, r, "k", nil)
	y := l.N.Name
	l.Release()
	if y == x {
		t.Fatal("a key must leave a node that is down")
	}
	nodeByName(r, x).SetUp(true)
	l = acquire(t, r, "k", nil)
	defer l.Release()
	if l.N.Name != y {
		t.Fatalf("the key went back to %s after recovery, want it to stay on %s", l.N.Name, y)
	}
	if placed(r, x) != 0 || placed(r, y) != 1 {
		t.Fatalf("placement counts wrong after the move: %+v", r.Snapshot())
	}
}

func TestPlacementSurvivesARetryOnAnotherNode(t *testing.T) {
	r := newRouter(3, 4, placementOpts())
	l := acquire(t, r, "k", nil)
	x := l.N.Name
	l.Release()
	l = acquire(t, r, "k", map[string]struct{}{x: {}}) // the attempt on x failed and is retried elsewhere
	if l.N.Name == x {
		t.Fatal("the retry must avoid the excluded node")
	}
	l.Release()
	l = acquire(t, r, "k", nil)
	defer l.Release()
	if l.N.Name != x || placed(r, x) != 1 {
		t.Fatalf("a retry must not move the placement: back on %s, counts %+v", l.N.Name, r.Snapshot())
	}
}

func TestPlacementIsNotMovedByASpill(t *testing.T) {
	r := newRouter(2, 1, placementOpts())
	first := acquire(t, r, "k", nil)
	x := first.N.Name
	spilled := acquire(t, r, "k", nil) // x is full, so this request spills to the other node
	if spilled.N.Name == x {
		t.Fatal("expected a spill")
	}
	first.Release()
	spilled.Release()
	l := acquire(t, r, "k", nil)
	defer l.Release()
	if l.N.Name != x || placed(r, x) != 1 {
		t.Fatalf("after a spill the key must stay on %s: got %s, counts %+v", x, l.N.Name, r.Snapshot())
	}
}

func TestPlacementForgetsTheLeastRecentlyUsedPrefix(t *testing.T) {
	o := opts()
	o.PlacementSize = 2
	r := newRouter(3, 4, o)
	for _, key := range []string{"a", "b", "c", "a", "d"} {
		l := acquire(t, r, key, nil)
		l.Release()
		total := 0
		for _, s := range r.Snapshot() {
			total += s.Placed
		}
		if total > 2 {
			t.Fatalf("%d prefixes remembered after %q, limit is 2", total, key)
		}
	}
}

func TestPlacementWorksWithTheAffinityWait(t *testing.T) {
	o := placementOpts()
	o.AffinityWait = 500 * time.Millisecond
	o.QueueTimeout = 2 * time.Second
	r := newRouter(2, 1, o)
	first := acquire(t, r, "k", nil)
	x := first.N.Name
	got := make(chan string, 1)
	go func() {
		l, err := r.Acquire(context.Background(), "k", "", nil, nil)
		if err != nil {
			got <- "error: " + err.Error()
			return
		}
		got <- l.N.Name
		l.Release()
	}()
	time.Sleep(60 * time.Millisecond)
	first.Release()
	select {
	case n := <-got:
		if n != x {
			t.Fatalf("waiting request went to %s, want the placed node %s", n, x)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("waiting request never got a node")
	}
}
