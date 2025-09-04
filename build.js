const fs = require('fs');
const fsp = require('fs').promises;
const path = require('path');
const matter = require('gray-matter');
const { marked } = require('marked');
const mkdirp = require('mkdirp');

const POSTS_DIR = path.join(__dirname, 'posts');
const DIST_DIR = path.join(__dirname, 'docs');
const POSTS_DIST = path.join(DIST_DIR, 'posts');
const ASSETS_SRC = path.join(__dirname, 'assets');
const ASSETS_DIST = path.join(DIST_DIR, 'assets');

const siteMeta = {
  title: "Goku's Blog",
  author: 'Goku',
  baseUrl: 'https://sameerdev7.github.io/',
  description: 'A professional blog about machine learning experiments, programming tutorials, and development insights.',
  twitter: '@_sammeeer',
  ogImage: 'https://sameerdev7.github.io/assets/son_kun2.jpg'
};

const htmlEscape = (s = '') =>
  String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// Function to generate meta tags
function generateMetaTags(post = null) {
  const isPost = post !== null;
  const title = isPost ? `${post.title} - ${siteMeta.title}` : siteMeta.title;
  const description = isPost ? post.excerpt : siteMeta.description;
  const url = isPost ? `${siteMeta.baseUrl}/posts/${post.slug}.html` : siteMeta.baseUrl;
  const image = isPost && post.image ? `${siteMeta.baseUrl}/${post.image}` : siteMeta.ogImage;
  const type = isPost ? 'article' : 'website';
  
  return `
    <!-- Essential Meta Tags -->
    <meta property="og:title" content="${htmlEscape(title)}">
    <meta property="og:description" content="${htmlEscape(description)}">
    <meta property="og:image" content="${image}">
    <meta property="og:url" content="${url}">
    <meta property="og:type" content="${type}">
    <meta property="og:site_name" content="${htmlEscape(siteMeta.title)}">
    
    <!-- Twitter Card Tags -->
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="${htmlEscape(title)}">
    <meta name="twitter:description" content="${htmlEscape(description)}">
    <meta name="twitter:image" content="${image}">
    <meta name="twitter:creator" content="${siteMeta.twitter}">
    
    <!-- Additional Meta Tags -->
    <meta name="description" content="${htmlEscape(description)}">
    <meta name="author" content="${htmlEscape(siteMeta.author)}">
    ${isPost ? `
    <!-- Article specific meta -->
    <meta property="article:author" content="${htmlEscape(siteMeta.author)}">
    <meta property="article:published_time" content="${new Date(post.date).toISOString()}">
    ${post.tags && post.tags.length > 0 ? post.tags.map(tag => `<meta property="article:tag" content="${htmlEscape(tag)}">`).join('\n    ') : ''}
    ` : ''}
    
    <!-- Favicon -->
    <link rel="icon" type="image/png" href="${isPost ? '../' : ''}assets/favicon.png">`;
}

// Function to calculate reading time
function calculateReadingTime(content) {
  const wordsPerMinute = 200;
  const words = content.trim().split(/\s+/).length;
  const minutes = Math.ceil(words / wordsPerMinute);
  return minutes;
}

// Professional TOC generation
function generateProfessionalTOC(html) {
  const headings = [];
  const headingRegex = /<h([2-6])[^>]*>(.*?)<\/h[2-6]>/gi;
  let match;
  
  while ((match = headingRegex.exec(html)) !== null) {
    const level = parseInt(match[1]);
    const text = match[2].replace(/<[^>]*>/g, '');
    const id = text.toLowerCase()
      .replace(/[^\w\s-]/g, '')
      .replace(/\s+/g, '-')
      .replace(/-+/g, '-')
      .trim();
    
    headings.push({ level, text, id });
  }
  
  if (headings.length === 0) return { toc: '', modifiedHtml: html };
  
  // Add IDs to headings in HTML
  let modifiedHtml = html;
  headings.forEach(heading => {
    const originalHeading = `<h${heading.level}>${heading.text}</h${heading.level}>`;
    const newHeading = `<h${heading.level} id="${heading.id}">${heading.text}</h${heading.level}>`;
    modifiedHtml = modifiedHtml.replace(originalHeading, newHeading);
  });
  
  // Group headings by main sections (H2) and subsections (H3+)
  const sections = [];
  let currentSection = null;
  
  headings.forEach((heading, index) => {
    if (heading.level === 2) {
      if (currentSection) sections.push(currentSection);
      currentSection = {
        main: heading,
        subsections: []
      };
    } else if (heading.level >= 3 && currentSection) {
      currentSection.subsections.push(heading);
    }
  });
  if (currentSection) sections.push(currentSection);
  
  // Generate professional TOC HTML
  const toc = `
    <div class="toc-professional">
      <h3>On this page</h3>
      <ul>
        ${sections.map(section => `
          <li>
            <a href="#${section.main.id}">
              ${htmlEscape(section.main.text)}
            </a>
            ${section.subsections.length > 0 ? `
              <ul>
                ${section.subsections.map(sub => `
                  <li>
                    <a href="#${sub.id}">
                      ${htmlEscape(sub.text)}
                    </a>
                  </li>
                `).join('')}
              </ul>
            ` : ''}
          </li>
        `).join('')}
      </ul>
    </div>
  `;
  
  return { toc, modifiedHtml };
}

async function copyAssets(srcDir, outDir) {
  try {
    const entries = await fsp.readdir(srcDir, { withFileTypes: true });
    await mkdirp(outDir);
    for (const e of entries) {
      const srcPath = path.join(srcDir, e.name);
      const destPath = path.join(outDir, e.name);
      if (e.isDirectory()) {
        await copyAssets(srcPath, destPath);
      } else {
        await fsp.copyFile(srcPath, destPath);
      }
    }
  } catch (err) {
    console.log('Assets directory not found or empty, skipping...');
  }
}

async function readPosts() {
  try {
    const files = (await fsp.readdir(POSTS_DIR)).filter(f => f.endsWith('.md'));
    console.log(`Found ${files.length} markdown files:`, files);
    
    const posts = [];
    for (const file of files) {
      try {
        const raw = await fsp.readFile(path.join(POSTS_DIR, file), 'utf8');
        const { data, content } = matter(raw);
        const slug = data.slug || file.replace(/\.md$/, '');
        const html = marked.parse(content);
        
        // Generate professional TOC
        const { toc, modifiedHtml } = generateProfessionalTOC(html);
        
        // Calculate reading time
        const readingTime = calculateReadingTime(content);
        
        posts.push({
          title: data.title || slug,
          date: data.date || '1970-01-01',
          excerpt:
            data.excerpt ||
            (content.split('\n').find(l => l.trim()) || '').slice(0, 160),
          image: data.image || '',
          tags: data.tags || [],
          slug,
          html: modifiedHtml,
          toc,
          readingTime
        });
        console.log(`Processed: ${file} -> ${slug}`);
      } catch (err) {
        console.error(`Error processing ${file}:`, err.message);
      }
    }
    posts.sort((a, b) => new Date(b.date) - new Date(a.date));
    return posts;
  } catch (err) {
    console.error('Error reading posts directory:', err.message);
    return [];
  }
}

function renderIndex(posts) {
  const postCards = posts
    .map(
      p => `
        <article class="post-card">
          <a href="posts/${p.slug}.html">
            ${p.image ? `
              <img src="${p.image}" alt="${htmlEscape(p.title)}" class="post-card-image">
            ` : ''}
            <div class="post-card-content">
              <div class="post-meta">
                ${htmlEscape(new Date(p.date).toLocaleDateString())} • ${p.readingTime} min read
              </div>
              <h3 class="post-card-title">${htmlEscape(p.title)}</h3>
              <p class="post-excerpt line-clamp-3">${htmlEscape(p.excerpt)}</p>
              ${p.tags && p.tags.length > 0 ? `
                <div class="mt-3">
                  ${p.tags.map(tag => `
                    <span class="tag">
                      ${htmlEscape(tag)}
                    </span>
                  `).join('')}
                </div>
              ` : ''}
            </div>
          </a>
        </article>`
    )
    .join('\n        ');

  const templatePath = path.join(__dirname, 'index.html');
  let template = fs.readFileSync(templatePath, 'utf8');
  return template.replace('<!-- POST_CARDS -->', postCards);
}

function renderPostTemplate(post) {
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>${htmlEscape(post.title)} - Goku's Blog</title>

${generateMetaTags(post)}

<script src="https://cdn.tailwindcss.com"></script>
<link href="https://cdn.jsdelivr.net/npm/prismjs/themes/prism-tomorrow.min.css" rel="stylesheet" />

<!-- Fixed MathJax Configuration -->
<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
    ignoreHtmlClass: 'tex2jax_ignore',
    processHtmlClass: 'tex2jax_process'
  },
  svg: {
    fontCache: 'global'
  }
};

// Theme functionality
function initTheme() {
  const savedTheme = localStorage.getItem('theme') || 'light';
  document.documentElement.setAttribute('data-theme', savedTheme);
  updateThemeIcon(savedTheme);
}

function toggleTheme() {
  const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
  const newTheme = currentTheme === 'light' ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', newTheme);
  localStorage.setItem('theme', newTheme);
  updateThemeIcon(newTheme);
}

function updateThemeIcon(theme) {
  const icon = document.querySelector('.theme-icon');
  if (icon) {
    icon.textContent = theme === 'light' ? '🌙' : '☀️';
  }
}

document.addEventListener('DOMContentLoaded', initTheme);
</script>
<link rel="stylesheet" href="../style.css">
</head>
<body class="font-sans flex flex-col min-h-screen">
  <header class="flex justify-between items-center px-8 py-6 border-b">
    <a href="../index.html" class="flex items-center hover:opacity-80 transition-opacity">
      <span class="text-xl font-bold">Goku</span>
    </a>
    <div class="flex items-center space-x-6">
      <nav class="flex space-x-6 text-sm">
        <a href="../index.html" class="nav-link">Home</a>
        <a href="../about.html" class="nav-link">About</a>
        <a href="../skills.html" class="nav-link">Skills</a>
        <a href="../contact.html" class="nav-link">Contact</a>
      </nav>
      <button onclick="toggleTheme()" class="theme-toggle" title="Toggle theme">
        <span class="theme-icon">🌙</span>
      </button>
    </div>
  </header>

  <main class="max-w-4xl mx-auto py-8 px-8 flex-grow">
    <article class="max-w-3xl mx-auto">
      <header class="mb-8">
        <div class="flex items-center justify-between mb-4 text-secondary">
          <time class="text-sm">${htmlEscape(
            new Date(post.date).toLocaleDateString()
          )}</time>
          <span class="text-sm">${post.readingTime} min read</span>
        </div>
        
        <h1 class="text-4xl font-bold mb-4">${htmlEscape(post.title)}</h1>
        
        ${post.tags && post.tags.length > 0 ? `
          <div class="flex flex-wrap gap-2 mb-6">
            ${post.tags.map(tag => `
              <span class="tag">
                ${htmlEscape(tag)}
              </span>
            `).join('')}
          </div>
        ` : ''}
        
        ${post.image ? `<img src="../${post.image}" alt="${htmlEscape(post.title)}" class="post-feature-img"/>` : ''}
      </header>
      
      ${post.toc}
      
      <div class="prose">
        ${post.html}
      </div>
    </article>
  </main>

  <footer class="text-center text-secondary text-sm py-6 border-t">
    <div>&copy; ${new Date().getFullYear()} ${htmlEscape(siteMeta.title)}</div>
  </footer>

  <script src="https://cdn.jsdelivr.net/npm/prismjs/prism.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/prismjs/components/prism-python.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/prismjs/components/prism-javascript.min.js"></script>
  <script>Prism.highlightAll();</script>
  
  <!-- Fixed MathJax Script -->
  <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</body>
</html>`;
}

async function writeStaticFiles(posts) {
  await mkdirp(POSTS_DIST);
  
  console.log(`Generating index.html with ${posts.length} posts...`);
  await fsp.writeFile(
    path.join(DIST_DIR, 'index.html'),
    renderIndex(posts),
    'utf8'
  );

  const staticFiles = ['about.html', 'skills.html', 'contact.html', 'style.css'];
  for (const f of staticFiles) {
    const src = path.join(__dirname, f);
    try {
      const data = await fsp.readFile(src, 'utf8');
      await fsp.writeFile(path.join(DIST_DIR, f), data, 'utf8');
      console.log(`Copied: ${f}`);
    } catch (err) {
      console.log(`Skipped missing file: ${f}`);
    }
  }

  await copyAssets(ASSETS_SRC, ASSETS_DIST);

  for (const post of posts) {
    const outPath = path.join(POSTS_DIST, `${post.slug}.html`);
    await fsp.writeFile(outPath, renderPostTemplate(post), 'utf8');
    console.log(`Generated: posts/${post.slug}.html`);
  }
}

(async () => {
  try {
    console.log('🧹 Cleaning dist directory...');
    await fsp.rm(DIST_DIR, { recursive: true, force: true });
    await mkdirp(DIST_DIR);
    
    console.log('📖 Reading posts...');
    const posts = await readPosts();
    
    console.log('🏗️ Building site...');
    await writeStaticFiles(posts);
    
    console.log(`✅ Site generated successfully with ${posts.length} posts!`);
    console.log('📂 Open docs/index.html to view your blog');
  } catch (err) {
    console.error('❌ Build failed:', err);
    process.exit(1);
  }
})();