package router

import (
	"context"
	"io"
	"log"
	"net/http"
	"sync"
	"time"
)

// HealthOptions configures active health checks.
type HealthOptions struct {
	Interval           time.Duration
	Timeout            time.Duration
	Path               string
	UnhealthyThreshold int
	HealthyThreshold   int
}

type probeState struct{ fails, oks int }

// StartHealthChecks probes every node until ctx ends. A node is marked down after
// UnhealthyThreshold consecutive failures and up again after HealthyThreshold successes.
func (r *Router) StartHealthChecks(ctx context.Context, h HealthOptions, client *http.Client) {
	go func() {
		state := map[string]*probeState{}
		r.ProbeAll(ctx, h, client, state)
		t := time.NewTicker(h.Interval)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				r.ProbeAll(ctx, h, client, state)
			}
		}
	}()
}

// ProbeAll runs one round synchronously. state carries counters between rounds.
func (r *Router) ProbeAll(ctx context.Context, h HealthOptions, client *http.Client, state map[string]*probeState) {
	var wg sync.WaitGroup
	var mu sync.Mutex
	for _, n := range r.nodes {
		wg.Add(1)
		go func(n *Node) {
			defer wg.Done()
			ok := probe(ctx, client, n.URL+h.Path, h.Timeout)
			if ctx.Err() != nil {
				return // shutting down: an aborted probe says nothing about the node
			}
			mu.Lock()
			ps := state[n.Name]
			if ps == nil {
				ps = &probeState{}
				state[n.Name] = ps
			}
			if ok {
				ps.oks++
				ps.fails = 0
			} else {
				ps.fails++
				ps.oks = 0
			}
			up, down := ok && ps.oks >= h.HealthyThreshold, !ok && ps.fails >= h.UnhealthyThreshold
			mu.Unlock()
			switch {
			case up:
				if n.SetUp(true) {
					log.Printf("[Router] node %s marked up", n.Name)
				}
			case down:
				if n.SetUp(false) {
					log.Printf("[Router] node %s marked down by health check", n.Name)
				}
			}
		}(n)
	}
	wg.Wait()
}

// NewProbeState returns the counter map ProbeAll expects.
func NewProbeState() map[string]*probeState { return map[string]*probeState{} }

func probe(ctx context.Context, client *http.Client, url string, timeout time.Duration) bool {
	pctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(pctx, http.MethodGet, url, nil)
	if err != nil {
		return false
	}
	resp, err := client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 1024))
	return resp.StatusCode >= 200 && resp.StatusCode < 300
}
