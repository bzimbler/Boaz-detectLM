(() => {
  const navLinks = document.querySelectorAll(".nav-links a");
  const sections = [...document.querySelectorAll("main section[id]")];
  const menuToggle = document.querySelector(".menu-toggle");
  const nav = document.querySelector(".nav-links");

  menuToggle?.addEventListener("click", () => {
    nav?.classList.toggle("open");
  });

  navLinks.forEach((link) => {
    link.addEventListener("click", () => nav?.classList.remove("open"));
  });

  const setActiveNav = () => {
    const y = window.scrollY + 96;
    let current = sections[0]?.id;
    for (const section of sections) {
      if (section.offsetTop <= y) current = section.id;
    }
    navLinks.forEach((link) => {
      link.classList.toggle(
        "active",
        link.getAttribute("href") === `#${current}`
      );
    });
  };

  window.addEventListener("scroll", setActiveNav, { passive: true });
  setActiveNav();

  // Reveal on scroll
  const reveals = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("in");
            io.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12, rootMargin: "0px 0px -40px 0px" }
    );
    reveals.forEach((el) => io.observe(el));
  } else {
    reveals.forEach((el) => el.classList.add("in"));
  }

  // Run-path tabs
  const tabs = document.querySelectorAll(".path-tab");
  const panels = document.querySelectorAll(".path-panel");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const id = tab.dataset.path;
      tabs.forEach((t) => t.classList.toggle("active", t === tab));
      panels.forEach((p) =>
        p.classList.toggle("active", p.dataset.path === id)
      );
    });
  });

  // Code explorer
  const explorerBtns = document.querySelectorAll(".explorer-nav button");
  const explorerPanels = document.querySelectorAll(".explorer-panel");
  explorerBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = btn.dataset.topic;
      explorerBtns.forEach((b) => b.classList.toggle("active", b === btn));
      explorerPanels.forEach((p) =>
        p.classList.toggle("active", p.dataset.topic === id)
      );
    });
  });
})();
