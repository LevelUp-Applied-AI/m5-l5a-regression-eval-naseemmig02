/* ===========================
   NASEEM MIGDADI PORTFOLIO
   main.js
=========================== */

// ====== MODULE PROJECTS DATA ======
const repos = [
  { url: "https://github.com/LevelUp-Applied-AI/python-env-naseemmig02", name: "Python Environment Setup" },
  { url: "https://github.com/LevelUp-Applied-AI/levelup-applied-ai-aispire-prework-python-env-aispire-prework-python-env", name: "AiSpire Python Env Prework" },
  { url: "https://github.com/LevelUp-Applied-AI/prework-day-1-and-2-naseemmig02", name: "Prework Days 1 & 2" },

  { url: "https://github.com/LevelUp-Applied-AI/m1-l1-git-workflows-naseemmig02", name: "M1-L1: Git Workflows (Lecture)" },
  { url: "https://github.com/LevelUp-Applied-AI/m1-d1-git-workflows-naseemmig02", name: "M1-D1: Git Workflows (Demo)" },
  { url: "https://github.com/LevelUp-Applied-AI/engineering-naseemmig02", name: "M1: Engineering Fundamentals" },

  { url: "https://github.com/LevelUp-Applied-AI/m2-l2-data-pipeline-naseemmig02", name: "M2-L2: Data Pipeline (Lecture)" },
  { url: "https://github.com/LevelUp-Applied-AI/m2-d2-pipeline-basics-naseemmig02", name: "M2-D2: Pipeline Basics (Demo)" },
  { url: "https://github.com/LevelUp-Applied-AI/m2-i2-pytorch-naseemmig02", name: "M2-I2: PyTorch Deep Learning" },
  { url: "https://github.com/LevelUp-Applied-AI/levelup-applied-ai-aispire-14005-m2-i2-pytorch-aispire-m2-i2-pytorch", name: "AiSpire M2-I2: PyTorch" },

  { url: "https://github.com/LevelUp-Applied-AI/m3-l3-sql-analytics-naseemmig02", name: "M3-L3: SQL Analytics (Lecture)" },
  { url: "https://github.com/LevelUp-Applied-AI/m3-d3-sql-fundamentals-naseemmig02", name: "M3-D3: SQL Fundamentals (Demo)" },
  { url: "https://github.com/LevelUp-Applied-AI/m3-i3-etl-pipeline-naseemmig02", name: "M3-I3: ETL Pipeline" },
  { url: "https://github.com/LevelUp-Applied-AI/m3-s2-time-series-analytics-naseemmig02", name: "M3-S2: Time Series Analytics" },
  { url: "https://github.com/naseemmig02/SQL-Murder-Mystery", name: "SQL Murder Mystery" },

  { url: "https://github.com/LevelUp-Applied-AI/m4-l4-descriptive-analytics-naseemmig02", name: "M4-L4: Descriptive Analytics (Lecture)" },
  { url: "https://github.com/LevelUp-Applied-AI/m4-d4-descriptive-analytics-naseemmig02-1", name: "M4-D4: Descriptive Analytics (Demo)" },
  { url: "https://github.com/LevelUp-Applied-AI/m4-i4-kpi-dashboard-naseemmig02", name: "M4-I4: KPI Dashboard" },
  { url: "https://github.com/LevelUp-Applied-AI/m4-s2", name: "M4-S2: Analytics Sprint" },

  { url: "https://github.com/LevelUp-Applied-AI/m5-d5a-classification-basics-naseemmig02", name: "M5-D5A: Classification Basics" },
  { url: "https://github.com/LevelUp-Applied-AI/m5-l5a-regression-eval-naseemmig02", name: "M5-L5A: Regression & Evaluation" },
];

// ====== MODULE GROUP DEFINITIONS ======
const moduleGroups = [
  {
    num: "Prework",
    title: "Environment & Foundations",
    urls: [
      "https://github.com/LevelUp-Applied-AI/python-env-naseemmig02",
      "https://github.com/LevelUp-Applied-AI/levelup-applied-ai-aispire-prework-python-env-aispire-prework-python-env",
      "https://github.com/LevelUp-Applied-AI/prework-day-1-and-2-naseemmig02",
    ]
  },
  {
    num: "Module 1",
    title: "Git & Engineering Workflows",
    urls: [
      "https://github.com/LevelUp-Applied-AI/m1-l1-git-workflows-naseemmig02",
      "https://github.com/LevelUp-Applied-AI/m1-d1-git-workflows-naseemmig02",
      "https://github.com/LevelUp-Applied-AI/engineering-naseemmig02",
    ]
  },
  {
    num: "Module 2",
    title: "Data Pipelines & Deep Learning (PyTorch)",
    urls: [
      "https://github.com/LevelUp-Applied-AI/m2-l2-data-pipeline-naseemmig02",
      "https://github.com/LevelUp-Applied-AI/m2-d2-pipeline-basics-naseemmig02",
      "https://github.com/LevelUp-Applied-AI/m2-i2-pytorch-naseemmig02",
      "https://github.com/LevelUp-Applied-AI/levelup-applied-ai-aispire-14005-m2-i2-pytorch-aispire-m2-i2-pytorch",
    ]
  },
  {
    num: "Module 3",
    title: "SQL Analytics, ETL & Time Series",
    urls: [
      "https://github.com/LevelUp-Applied-AI/m3-l3-sql-analytics-naseemmig02",
      "https://github.com/LevelUp-Applied-AI/m3-d3-sql-fundamentals-naseemmig02",
      "https://github.com/LevelUp-Applied-AI/m3-i3-etl-pipeline-naseemmig02",
      "https://github.com/LevelUp-Applied-AI/m3-s2-time-series-analytics-naseemmig02",
      "https://github.com/naseemmig02/SQL-Murder-Mystery",
    ]
  },
  {
    num: "Module 4",
    title: "Descriptive Analytics & KPI Dashboards",
    urls: [
      "https://github.com/LevelUp-Applied-AI/m4-l4-descriptive-analytics-naseemmig02",
      "https://github.com/LevelUp-Applied-AI/m4-d4-descriptive-analytics-naseemmig02-1",
      "https://github.com/LevelUp-Applied-AI/m4-i4-kpi-dashboard-naseemmig02",
      "https://github.com/LevelUp-Applied-AI/m4-s2",
    ]
  },
  {
    num: "Module 5",
    title: "Machine Learning — Classification & Regression",
    urls: [
      "https://github.com/LevelUp-Applied-AI/m5-d5a-classification-basics-naseemmig02",
      "https://github.com/LevelUp-Applied-AI/m5-l5a-regression-eval-naseemmig02",
    ]
  },
];

// ====== RENDER MODULE GROUPS ======
function renderModuleGroups() {
  const container = document.getElementById('module-groups');
  if (!container) return;

  const repoMap = {};
  repos.forEach(r => { repoMap[r.url] = r.name; });

  moduleGroups.forEach((group, gi) => {
    const groupEl = document.createElement('div');
    groupEl.className = 'module-group fade-in';
    groupEl.style.transitionDelay = (gi * 0.07) + 's';

    const header = document.createElement('div');
    header.className = 'module-group-header';
    header.innerHTML = `
      <span class="module-num">${group.num}</span>
      <span class="module-title">${group.title}</span>
    `;

    const cards = document.createElement('div');
    cards.className = 'module-cards';

    group.urls.forEach(url => {
      const name = repoMap[url] || url.split('/').pop();
      const card = document.createElement('a');
      card.className = 'module-card';
      card.href = url;
      card.target = '_blank';
      card.rel = 'noopener noreferrer';
      card.innerHTML = `
        <span class="module-card-name">${name}</span>
        <span class="module-card-arrow">↗</span>
      `;
      cards.appendChild(card);
    });

    groupEl.appendChild(header);
    groupEl.appendChild(cards);
    container.appendChild(groupEl);
  });
}

// ====== NAV SCROLL EFFECT ======
function initNav() {
  const nav = document.getElementById('nav');
  window.addEventListener('scroll', () => {
    if (window.scrollY > 40) {
      nav.classList.add('scrolled');
    } else {
      nav.classList.remove('scrolled');
    }
  });

  // Active link highlighting
  const sections = ['home', 'about', 'experience', 'projects', 'contact'];
  const links = document.querySelectorAll('.nav-links a');

  window.addEventListener('scroll', () => {
    let current = '';
    sections.forEach(id => {
      const el = document.getElementById(id);
      if (el && window.scrollY >= el.offsetTop - 100) {
        current = id;
      }
    });
    links.forEach(link => {
      link.style.color = '';
      if (link.getAttribute('href') === '#' + current) {
        link.style.color = 'var(--accent)';
      }
    });
  });
}

// ====== INTERSECTION OBSERVER FOR FADE-INS ======
function initScrollAnimations() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
      }
    });
  }, { threshold: 0.1 });

  // Observe all fade-in elements (including dynamically added ones)
  function observe() {
    document.querySelectorAll('.fade-in').forEach(el => observer.observe(el));
  }
  observe();

  // Re-observe after module groups are rendered
  setTimeout(observe, 100);

  // Also animate section children
  const animatables = document.querySelectorAll(
    '.timeline-item, .stat-card, .contact-card, .featured-card, .skill-tag'
  );
  animatables.forEach((el, i) => {
    el.classList.add('fade-in');
    el.style.transitionDelay = (i % 6) * 0.07 + 's';
    observer.observe(el);
  });
}

// ====== INIT ======
document.addEventListener('DOMContentLoaded', () => {
  renderModuleGroups();
  initNav();
  initScrollAnimations();
});
