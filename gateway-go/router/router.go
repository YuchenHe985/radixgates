// Package router implements prefix-hash consistent routing to SGLang instances.
// Each node has a semaphore that bounds its concurrency without rate-limiting
// the caller — excess goroutines simply wait until a slot opens.
package router

import (
	"context"
	"fmt"
	"hash/fnv"
	"log"

	"github.com/unicoregpu/radixgates/gateway/config"
)

// Node represents one SGLang inference server.
type Node struct {
	Group string
	Role  string // "" = combined | "prefill" = prefill-only | "decode" = decode-only
	URL   string
	sem   chan struct{} // bounded concurrency semaphore
}

// Acquire blocks until a concurrency slot is available or ctx is cancelled.
func (n *Node) Acquire(ctx context.Context) error {
	select {
	case n.sem <- struct{}{}:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Release frees one concurrency slot.
func (n *Node) Release() {
	<-n.sem
}

// ── Router ───────────────────────────────────────────────────────────────────

// Router holds the pool of SGLang nodes and performs prefix-hash routing.
type Router struct {
	nodes []*Node
}

// New builds a Router from the gateway config.
func New(instances []config.SGLangInstance, defaultMaxConcurrent int) *Router {
	nodes := make([]*Node, 0, len(instances))
	for _, inst := range instances {
		mc := inst.MaxConcurrent
		if mc <= 0 {
			mc = defaultMaxConcurrent
		}
		n := &Node{
			Group: inst.Group,
			Role:  inst.Role,
			URL:   fmt.Sprintf("http://%s:%d", inst.Host, inst.Port),
			sem:   make(chan struct{}, mc),
		}
		log.Printf("[Router] Node registered: group=%q role=%q url=%s max_concurrent=%d",
			n.Group, n.Role, n.URL, mc)
		nodes = append(nodes, n)
	}
	return &Router{nodes: nodes}
}

// Select picks a Node for normal (non-PD) routing using consistent hashing.
// Prefill-only nodes are excluded — they are not valid request entry points.
// If targetGroup is non-empty, only nodes in that group are considered.
func (r *Router) Select(prefixHash, targetGroup string) (*Node, bool) {
	return r.selectFrom(prefixHash, targetGroup, func(n *Node) bool {
		return n.Role != "prefill" // skip prefill-only nodes
	})
}

// SelectPD picks a decode-role Node for PD-disaggregated routing.
// In SGLang's PD design, the decode node is the request entry point:
// it bootstraps the prefill node, receives the KV cache, and generates tokens.
// Falls back to combined-role nodes if no decode-only node exists.
func (r *Router) SelectPD(prefixHash, targetGroup string) (*Node, bool) {
	// Try decode-role nodes first
	if n, ok := r.selectFrom(prefixHash, targetGroup, func(n *Node) bool {
		return n.Role == "decode"
	}); ok {
		return n, true
	}
	// Fall back to combined nodes (no PD setup, graceful degradation)
	return r.selectFrom(prefixHash, targetGroup, func(n *Node) bool {
		return n.Role == ""
	})
}

// selectFrom is the internal consistent-hash implementation with a filter.
func (r *Router) selectFrom(prefixHash, targetGroup string, filter func(*Node) bool) (*Node, bool) {
	pool := make([]*Node, 0, len(r.nodes))
	for _, n := range r.nodes {
		if targetGroup != "" && n.Group != targetGroup {
			continue
		}
		if filter(n) {
			pool = append(pool, n)
		}
	}
	if len(pool) == 0 {
		return nil, false
	}
	// FNV-32 hash of prefixHash → deterministic index
	h := fnv.New32a()
	_, _ = h.Write([]byte(prefixHash))
	idx := int(h.Sum32()) % len(pool)
	return pool[idx], true
}
