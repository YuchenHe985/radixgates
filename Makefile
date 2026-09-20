.PHONY: test bench plots

test:
	cd gateway-go && go vet ./... && go test -race -count=1 ./...

bench:
	./benchmarks/scripts/run_sim.sh crash
	./benchmarks/scripts/run_sim.sh brownout

plots:
	python3 benchmarks/scripts/plot.py
