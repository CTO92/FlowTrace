.PHONY: help install check build-graph update-graph run-listener run-dashboard backtest export docker-up docker-down clean

help:
	@echo "Available commands:"
	@echo "  make install         - Install Python dependencies and browser binaries"
	@echo "  make check           - Check environment variables and dependencies"
	@echo "  make build-graph     - Initialize and seed the Knowledge Graph (Phase 1)"
	@echo "  make update-graph    - Update Graph with latest SEC filings"
	@echo "  make run-listener    - Start the ingestion listener (backend)"
	@echo "  make run-dashboard   - Start the Streamlit dashboard (frontend)"
	@echo "  make backtest        - Run historical backtest"
	@echo "  make export          - Export graph to Gephi/Pyvis"
	@echo "  make docker-up       - Build and start Docker containers"
	@echo "  make docker-down     - Stop Docker containers"
	@echo "  make clean           - Remove temporary files and databases"

install:
	pip install -r requirements.txt
	python -m playwright install chromium

check:
	python check_env.py
	python check_dependencies.py

build-graph:
	python build_knowledge_graph.py

update-graph:
	python update_knowledge_graph.py

run-listener:
	python ingestion_listener.py

run-dashboard:
	streamlit run app.py

backtest:
	python backtest.py

export:
	python export_graph.py

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

clean:
	# Note: These commands assume a Unix-like shell (Git Bash, WSL, Linux, Mac)
	rm -f knowledge_graph.db backtest_results.csv
	rm -rf chroma_db exports __pycache__