VENV_BIN=./bin

.PHONY: app main venv

venv:
	@test -d $(VENV_BIN) || python3 -m venv .
	@$(VENV_BIN)/pip install --upgrade pip | cat
	@$(VENV_BIN)/pip install -q -r requirements.txt 2>/dev/null || true

app:
	@source $(VENV_BIN)/activate && echo "" | streamlit run app.py

main:
	@source $(VENV_BIN)/activate && python main.py



