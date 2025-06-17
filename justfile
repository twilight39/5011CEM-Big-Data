help:
    just --list --justfile {{justfile()}}

notebook:
    #!/usr/bin/env bash
    find notebooks -name "*.py" | while read -r pyfile; do
        echo "Converting $pyfile to notebook..."
        python -m jupytext --to notebook "$pyfile"
    done
