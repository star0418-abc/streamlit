"""
更新报告 / Update Report - GPE Lab

Displays the version changelog for the application.
"""
import streamlit as st

# Idempotent path setup (avoids duplicate insertions on reruns)
import sys
from pathlib import Path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from utils.i18n import t, init_language, language_selector
from utils.changelog import read_changelog_markdown, filter_changelog_content

# Page configuration
st.set_page_config(
    page_title=t("changelog.page_title"),
    page_icon="📋",
    layout="wide"
)

# Initialize language
init_language()

# Sidebar
with st.sidebar:
    language_selector()

# Page content
st.title(t("changelog.title"))
st.markdown(t("changelog.subtitle"))

st.markdown("---")

# Search filter (optional enhancement)
search_term = st.text_input(
    t("changelog.search_placeholder"),
    key="changelog_search",
    label_visibility="collapsed",
    placeholder=t("changelog.search_placeholder")
)

# Read and display changelog
success, content = read_changelog_markdown()

if success:
    if search_term:
        filtered_content = filter_changelog_content(content, search_term)
        st.markdown(filtered_content)
    else:
        st.markdown(content)
else:
    st.warning(t("changelog.no_changelog"))
    st.info("""
**管理员提示 / Admin Note:**

要添加更新记录，请创建 `data/changelog_zh.md` 文件，格式如下：

```markdown
## 2026-01-31 v0.1.1

- 新增：功能描述
- 修复：修复内容
- 改进：改进内容
```
    """)
