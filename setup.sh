mkdir -p ~/.streamlit/

echo "\n\
[general]\n\
email = \"dieter12387@googlemail.com\"\n\
" >> ~/.streamlit/credentials.toml

echo "\n\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = $PORT\n\
" >> ~/.streamlit/config.toml
