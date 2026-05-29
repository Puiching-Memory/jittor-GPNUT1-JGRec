(function () {
  function renderMermaid() {
    if (!window.mermaid) {
      return;
    }

    var diagrams = document.querySelectorAll(".mermaid:not([data-processed])");
    if (!diagrams.length) {
      return;
    }

    var scheme = document.body.getAttribute("data-md-color-scheme");
    window.mermaid.initialize({
      startOnLoad: false,
      theme: scheme === "slate" ? "dark" : "default",
    });

    window.mermaid.run({ querySelector: ".mermaid" }).catch(function (error) {
      console.error("Failed to render Mermaid diagrams", error);
    });
  }

  if (window.document$) {
    window.document$.subscribe(renderMermaid);
  } else {
    document.addEventListener("DOMContentLoaded", renderMermaid);
  }

  window.addEventListener("load", renderMermaid);
})();
