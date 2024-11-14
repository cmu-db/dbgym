#!/bin/bash
# You may need to do `pkill python` to fully restart the streamlit server. If you do not do this, objects cached
# with @st.cache_resource may still be persisted even after you do Ctrl-C and rerun ./scripts/run_demo.sh.
python -m streamlit run tune/demo/main.py