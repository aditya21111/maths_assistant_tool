# Math Assistant — AI-Powered Math Problem Solver

LLM-powered math assistant built with **LangChain Agents**, **Streamlit**, and **SymPy**. Converts natural-language math questions into tool calls, solves them symbolically/numerically, and streams the answer back.

## Supported Problem Types

| Category | Examples |
|----------|---------|
| Arithmetic | `what is 55!`, `sqrt(16) + 3` |
| Trigonometry | `value of sin90 deg`, `cos(pi/4)` |
| Single-variable equations | `solve 3x² + 5x + 2 = 0`, `x² + sin(x) = 1` |
| Systems of equations | `solve 2x - y + z = 3, x + y + z = 6` |
| Differentiation | `d/dx of x³·sin(x)`, higher-order & partial derivatives |
| Integration | Indefinite, definite, and multi-variable integrals |
| Limits | `limit of sin(x)/x as x→0`, one-sided limits |
| Series expansion | Taylor/Maclaurin series around a point |

## Tools

All tools live in [`math_tools.py`](math_tools.py):

| Tool | Engine | Purpose |
|------|--------|---------|
| `calculator` | numexpr | Arithmetic evaluation |
| `solve_one_variable_equations` | sympy.solve + nsolve | Single-variable equations |
| `solve_multi_variable_equations` | sympy.solve + nsolve | Systems of equations |
| `differentiate_expression` | sympy.diff | Derivatives (any order, partial) |
| `integration_nonnumeric` | sympy.integrate | Indefinite, definite, iterated integrals |
| `solve_limits` | sympy.limit | Limits including one-sided (±) |
| `find_series_expansion` | sympy.series | Taylor/Maclaurin expansion |

## Tech Stack

- **LLM**: Google Gemma 4 31B (`langchain-google-genai`)
- **Agent**: LangChain `create_agent` (LangGraph-based)
- **Symbolic math**: SymPy
- **Numeric**: numexpr + NumPy
- **UI**: Streamlit
- **Tracing & Evaluation**: LangSmith

## Setup

**Prerequisites:** Python ≥ 3.11, [uv](https://docs.astral.sh/uv/) or pip.

```bash
git clone <repo-url>
cd text-to-math

# Install (pick one)
uv sync
pip install -r requirements.txt

# Configure
cp .env.example .env
```

### Environment variables

| Variable | Required | Purpose |
|----------|:--------:|---------|
| `GOOGLE_API_KEY` | ✅ | Gemma model access |
| `LANGSMITH_API_KEY` | optional | LangSmith tracing & evaluation |
| `LANGSMITH_TRACING` | optional | Set to `true` to activate tracing (auto-set in code) |
| `LANGSMITH_PROJECT` | optional | Project name for grouping traces |

## Run

```bash
streamlit run main.py
```

## LangSmith Tracing

When `LANGSMITH_API_KEY` is set in `.env`, every chat interaction is automatically traced — agent reasoning, tool calls, and latency are all logged to your [LangSmith](https://smith.langchain.com/) dashboard.

## Evaluation

Click **"Run experiments"** in the sidebar to evaluate against a LangSmith dataset. Results download as CSV.

**Using your own dataset:**

1. Create a dataset on [smith.langchain.com](https://smith.langchain.com/) with `question` and `output` fields.
2. Replace the dataset name in `main.py`:
   ```python
   evaluation_result = evaluate(
       agent_predict,
       data='jee_math_hard_30',  # ← your dataset name
   )
   ```

## Project Structure

```
text-to-math/
├── main.py              # Streamlit app + agent + evaluation
├── math_tools.py        # All 7 math tools
├── experiments.ipynb     # Development notebook
├── requirements.txt      # Pinned dependencies
├── pyproject.toml        # Project metadata & dependencies
└── .env                  # API keys (not committed)
```

## Roadmap — Upcoming Tools

| Tool | Coverage | Status |
|------|----------|:------:|
| `matrix_tool` | Determinants, inverse, rank, eigenvalues, Ax=b | 🔜 |
| `vector_3d_tool` | Dot/cross product, projections, planes & lines in 3D | 🔜 |
| `combinatorics_tool` | Permutations, combinations, binomial expansion | 📋 |
| `probability_tool` | Bayes' theorem, conditional probability, distributions | 📋 |
| `coordinate_geometry_tool` | Conic properties, tangent/normal to conics | 📋 |

## License

MIT
