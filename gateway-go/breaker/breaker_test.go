package breaker

import (
	"sync"
	"testing"
	"time"
)

type fakeClock struct {
	mu sync.Mutex
	t  time.Time
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *fakeClock) Advance(d time.Duration) {
	c.mu.Lock()
	c.t = c.t.Add(d)
	c.mu.Unlock()
}

func newTest(cfg Config) (*Breaker, *fakeClock, *[]string) {
	clk := &fakeClock{t: time.Unix(1_700_000_000, 0)}
	var changes []string
	b := New(cfg, WithClock(clk.Now), WithOnChange(func(from, to State) {
		changes = append(changes, from.String()+">"+to.String())
	}))
	return b, clk, &changes
}

func failN(t *testing.T, b *Breaker, n int) {
	t.Helper()
	for i := 0; i < n; i++ {
		tk, ok := b.Allow()
		if !ok {
			t.Fatalf("Allow refused at failure %d (state %s)", i, b.State())
		}
		b.Done(tk, false)
	}
}

func TestTripsAfterConsecutiveFailures(t *testing.T) {
	b, _, changes := newTest(Config{FailureThreshold: 3, OpenDuration: time.Second, HalfOpenMax: 1})
	failN(t, b, 2)
	if b.State() != Closed {
		t.Fatalf("want closed after 2 failures, got %s", b.State())
	}
	failN(t, b, 1)
	if b.State() != Open {
		t.Fatalf("want open after 3 failures, got %s", b.State())
	}
	if _, ok := b.Allow(); ok {
		t.Fatal("Allow must refuse while open")
	}
	if len(*changes) != 1 || (*changes)[0] != "closed>open" {
		t.Fatalf("unexpected transitions %v", *changes)
	}
}

func TestSuccessResetsConsecutiveCount(t *testing.T) {
	b, _, _ := newTest(Config{FailureThreshold: 3, OpenDuration: time.Second, HalfOpenMax: 1})
	failN(t, b, 2)
	tk, _ := b.Allow()
	b.Done(tk, true)
	failN(t, b, 2)
	if b.State() != Closed {
		t.Fatalf("a success in between must reset the streak, state=%s", b.State())
	}
}

func TestHalfOpenProbeClosesOnSuccess(t *testing.T) {
	b, clk, changes := newTest(Config{FailureThreshold: 1, OpenDuration: time.Second, HalfOpenMax: 1})
	failN(t, b, 1)
	clk.Advance(999 * time.Millisecond)
	if b.State() != Open {
		t.Fatal("must stay open before the cool-down ends")
	}
	clk.Advance(2 * time.Millisecond)
	if b.State() != HalfOpen {
		t.Fatalf("want half_open after cool-down, got %s", b.State())
	}
	tk, ok := b.Allow()
	if !ok || !tk.probe {
		t.Fatal("first request in half-open must be admitted as a probe")
	}
	if b.Available() {
		t.Fatal("probe slots are exhausted, Available must be false")
	}
	if _, ok := b.Allow(); ok {
		t.Fatal("second concurrent probe must be refused when HalfOpenMax=1")
	}
	b.Done(tk, true)
	if b.State() != Closed {
		t.Fatalf("probe success must close the breaker, got %s", b.State())
	}
	want := []string{"closed>open", "open>half_open", "half_open>closed"}
	if len(*changes) != len(want) {
		t.Fatalf("transitions %v, want %v", *changes, want)
	}
	for i := range want {
		if (*changes)[i] != want[i] {
			t.Fatalf("transitions %v, want %v", *changes, want)
		}
	}
}

func TestProbeFailureReopensWithBackoff(t *testing.T) {
	b, clk, _ := newTest(Config{FailureThreshold: 1, OpenDuration: time.Second, MaxOpenDuration: 3 * time.Second, HalfOpenMax: 1})
	failN(t, b, 1)

	// probe 1 fails -> cool-down doubles to 2s
	clk.Advance(time.Second)
	tk, ok := b.Allow()
	if !ok {
		t.Fatal("probe must be admitted")
	}
	b.Done(tk, false)
	if b.State() != Open {
		t.Fatal("failed probe must reopen")
	}
	clk.Advance(1500 * time.Millisecond)
	if b.State() != Open {
		t.Fatal("cool-down should have doubled to 2s")
	}
	clk.Advance(600 * time.Millisecond)

	// probe 2 fails -> 4s capped at 3s
	tk, _ = b.Allow()
	b.Done(tk, false)
	clk.Advance(2900 * time.Millisecond)
	if b.State() != Open {
		t.Fatal("cool-down should be capped at 3s, not yet elapsed")
	}
	clk.Advance(200 * time.Millisecond)
	if b.State() != HalfOpen {
		t.Fatalf("want half_open after the capped cool-down, got %s", b.State())
	}

	// recovery resets the cool-down
	tk, _ = b.Allow()
	b.Done(tk, true)
	failN(t, b, 1)
	clk.Advance(time.Second)
	if b.State() != HalfOpen {
		t.Fatal("cool-down must reset to the base value after a successful close")
	}
}

func TestStaleResultsAreIgnored(t *testing.T) {
	b, clk, _ := newTest(Config{FailureThreshold: 1, OpenDuration: time.Second, HalfOpenMax: 1})
	slow, _ := b.Allow() // admitted while closed, still running

	failN(t, b, 1) // another request trips the breaker
	clk.Advance(time.Second)
	probe, ok := b.Allow()
	if !ok || !probe.probe {
		t.Fatal("expected a half-open probe")
	}

	b.Done(slow, true) // stale success must not close the breaker
	if b.State() != HalfOpen {
		t.Fatalf("stale success closed the breaker: %s", b.State())
	}
	b.Done(probe, true)
	if b.State() != Closed {
		t.Fatalf("real probe success must close, got %s", b.State())
	}
}

func TestCancelReturnsProbeSlot(t *testing.T) {
	b, clk, _ := newTest(Config{FailureThreshold: 1, OpenDuration: time.Second, HalfOpenMax: 1})
	failN(t, b, 1)
	clk.Advance(time.Second)
	tk, _ := b.Allow()
	if b.Available() {
		t.Fatal("slot should be taken")
	}
	b.Cancel(tk)
	if !b.Available() {
		t.Fatal("cancel must free the probe slot so the breaker cannot wedge in half_open")
	}
}

func TestConcurrentUse(t *testing.T) {
	b := New(Config{FailureThreshold: 5, OpenDuration: time.Millisecond, HalfOpenMax: 2})
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 2000; i++ {
				if tk, ok := b.Allow(); ok {
					b.Done(tk, (i+g)%3 != 0)
				}
				_ = b.State()
				_ = b.Available()
			}
		}(g)
	}
	wg.Wait() // meaningful under `go test -race`
}
