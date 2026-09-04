# Operator-only build. The measurement launcher never builds or pulls.
FROM node:22.19.0-bookworm-slim@sha256:4a4884e8a44826194dff92ba316264f392056cbe243dcc9fd3551e71cea02b90
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium=152.0.7977.75-1~deb12u1 chromium-common=152.0.7977.75-1~deb12u1 \
    python3=3.11.2-1+b1 ca-certificates fonts-liberation \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/audit-lighthouse
COPY src/ai_search_audit/lighthouse_assets/package.json src/ai_search_audit/lighthouse_assets/package-lock.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund \
    && install -d -o 1000 -g 1000 -m 0700 /run/audit-proxy \
    && mkdir ai_search_audit && touch ai_search_audit/__init__.py
COPY src/ai_search_audit/lighthouse_network.py src/ai_search_audit/lighthouse_proxy.py ./ai_search_audit/
COPY src/ai_search_audit/lighthouse_assets/runner.mjs src/ai_search_audit/lighthouse_assets/isolation.py src/ai_search_audit/lighthouse_assets/sidecar.py ./
COPY src/ai_search_audit/lighthouse_assets/LICENSE.moby ./LICENSE.moby
LABEL org.ai-search-audit.runner="0ed0f34c288df30fe74fc7be9a3bbbabe5ed0044431e4ca064c28b2adae69847"
LABEL org.ai-search-audit.seccomp="1fb1ca45bccc2b53163801a8e93758967230a87fac0e408a5d7c397bf77d7d17"
LABEL org.ai-search-audit.lock="a943a82330e7fe21f4a33427954da1aab7fce651f5de0eca7daa252f86575000"
LABEL org.ai-search-audit.isolation="148cf9c6a89a98f2e7a034d1dd784ceff84b88d86b7e8d11affeb84d5ff13e02"
LABEL org.ai-search-audit.sidecar="88a064eb00c6838793d6a19e466508be4a966780d230a38f1dd46221750483f2"
LABEL org.ai-search-audit.network="c026d7334fb49fcf71b944b08346087738a2e0ad0cbdf3c4cd714be780d9b03d"
LABEL org.ai-search-audit.proxy="98849be8e8ba48af79b430d8d22f42eb7435cab4cc1785416557f2ef293f7f34"
USER 1000:1000
ENTRYPOINT ["/usr/local/bin/node", "/opt/audit-lighthouse/runner.mjs"]
