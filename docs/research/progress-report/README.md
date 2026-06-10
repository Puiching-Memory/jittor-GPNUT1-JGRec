# 研究进度报告 LaTeX 源码

本目录使用 [`pmichaillat/latex-paper`](https://github.com/pmichaillat/latex-paper) 模板整理当前比赛项目研究进度报告。

## 文件

- `paper.tex`：报告正文。
- `paper.bib`：正文引用的论文条目。
- `paper.sty`、`paper.bst`：从模板仓库复制的样式文件。

## 编译

模板推荐使用 pdfTeX。由于正文为中文，当前源码使用 `CJKutf8` 包：

```bash
cd docs/research/progress-report
pdflatex paper
bibtex paper
pdflatex paper
pdflatex paper
```

如使用 TinyTeX，可能需要先安装模板字体和中文字体依赖：

```bash
tlmgr install sourceserif sourcecodepro mnsymbol mathalpha mathastext fourier \
  ly1 titling tocloft titlesec multirow caption enumitem natbib arphic
```
