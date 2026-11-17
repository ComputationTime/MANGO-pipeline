document.addEventListener("DOMContentLoaded", () => {
  const page = document.body.dataset.page;
  document.querySelectorAll("nav a[data-page]").forEach((link) => {
    if (link.dataset.page === page) link.setAttribute("aria-current", "page");
  });
});
