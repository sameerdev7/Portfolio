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
  title: "Sameer's Blog",
  author: 'Sameer',
  baseUrl: ''
};

const htmlEscape = (s = '') =>
  String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// Function to calculate reading time
function calculateReadingTime(content) {
  const wordsPerMinute = 200;
  const words = content.trim().split(/\s+/).length;
  const minutes = Math.ceil(words / wordsPerMinute);
  return minutes;
}

// OPTION 1: Manual TOC from frontmatter
function generateManualTOC(tocData) {
  if (!tocData || !Array.isArray(tocData)) return '';
  
  return `
    <div class="toc bg-gray-800 p-4 rounded-lg mb-6">
      <h3 class="text-lg font-semibold mb-3 text-teal-400">Table of Contents</h3>
      <ul class="space-y-1">
        ${tocData.map(item => `
          <li class="ml-${(item.level || 1) * 4}">
            <a href="#${item.id}" class="text-sm text-gray-300 hover:text-teal-400 transition-colors">
              ${htmlEscape(item.title)}
            </a>
          </li>
        `).join('')}
      </ul>
    </div>
  `;
}

// OPTION 2: Auto-generate but allow override
function generateTOC(html, manualToc = null) {
  // If manual TOC is provided, use it
  if (manualToc) {
    return { toc: generateManualTOC(manualToc), modifiedHtml: html };
  }
  
  // Otherwise auto-generate
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
  
  // Generate TOC HTML
  const toc = `
    <div class="toc bg-gray-800 p-4 rounded-lg mb-6">
      <h3 class="text-lg font-semibold mb-3 text-teal-400">Table of Contents</h3>
      <ul class="space-y-1">
        ${headings.map(h => `
          <li class="ml-${(h.level - 2) * 4}">
            <a href="#${h.id}" class="text-sm text-gray-300 hover:text-teal-400 transition-colors">
              ${htmlEscape(h.text)}
            </a>
          </li>
        `).join('')}
      </ul>
    </div>
  `;
  
  return { toc, modifiedHtml };
}

// Clean, numbered TOC like the design you showed
function generateCustomStyledTOC(html, customTocStyle = 'clean') {
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
  
  // Add IDs to headings
  let modifiedHtml = html;
  headings.forEach(heading => {
    const originalHeading = `<h${heading.level}>${heading.text}</h${heading.level}>`;
    const newHeading = `<h${heading.level} id="${heading.id}">${heading.text}</h${heading.level}>`;
    modifiedHtml = modifiedHtml.replace(originalHeading, newHeading);
  });
  
  let toc = '';
  
  // Group headings by main sections (H2) and subsections (H3+)
  const sections = [];
  let currentSection = null;
  
  headings.forEach((heading, index) => {
    if (heading.level === 2) {
      if (currentSection) sections.push(currentSection);
      currentSection = {
        main: heading,
        number: sections.length + 1,
        subsections: []
      };
    } else if (heading.level >= 3 && currentSection) {
      currentSection.subsections.push({
        ...heading,
        number: currentSection.subsections.length + 1
      });
    }
  });
  if (currentSection) sections.push(currentSection);
  
  switch (customTocStyle) {
    case 'clean':
      toc = `
        <div class="toc-clean bg-gray-50 border border-gray-200 rounded-lg p-6 mb-8" style="background: #1a202c; border-color: #2d3748;">
          <h3 class="text-lg font-semibold mb-4 text-gray-200">Table of Contents</h3>
          <ul class="space-y-2">
            ${sections.map(section => `
              <li>
                <a href="#${section.main.id}" class="text-blue-600 hover:text-blue-800 font-medium text-sm block" style="color: #4299e1;">
                  ${section.number}. ${htmlEscape(section.main.text)}
                </a>
                ${section.subsections.length > 0 ? `
                  <ul class="ml-6 mt-1 space-y-1">
                    ${section.subsections.map(sub => `
                      <li>
                        <a href="#${sub.id}" class="text-blue-500 hover:text-blue-700 text-sm" style="color: #63b3ed;">
                          ${section.number}.${sub.number} ${htmlEscape(sub.text)}
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
      break;
      
    case 'minimal':
      toc = `
        <nav class="toc-minimal border-l-2 border-teal-500 pl-4 mb-8">
          <p class="text-sm text-teal-400 font-semibold mb-2">Contents</p>
          ${headings.map(h => `
            <a href="#${h.id}" class="block text-sm text-gray-400 hover:text-teal-400 py-1 pl-${(h.level - 2) * 3}">
              ${htmlEscape(h.text)}
            </a>
          `).join('')}
        </nav>
      `;
      break;
      
    default: // 'default' 
      toc = `
        <div class="toc bg-gray-800 p-4 rounded-lg mb-6">
          <h3 class="text-lg font-semibold mb-3 text-teal-400">Table of Contents</h3>
          <ul class="space-y-1">
            ${headings.map(h => `
              <li class="ml-${(h.level - 2) * 4}">
                <a href="#${h.id}" class="text-sm text-gray-300 hover:text-teal-400 transition-colors">
                  ${htmlEscape(h.text)}
                </a>
              </li>
            `).join('')}
          </ul>
        </div>
      `;
  }
  
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
        
        // Generate TOC - use manual if provided, otherwise auto-generate
        const tocStyle = data.tocStyle || 'default'; // can be 'default', 'minimal', 'numbered', 'sidebar', or 'none'
        let toc = '';
        let modifiedHtml = html;
        
        if (tocStyle !== 'none') {
          if (data.toc) {
            // Manual TOC from frontmatter
            const result = generateTOC(html, data.toc);
            toc = result.toc;
            modifiedHtml = result.modifiedHtml;
          } else {
            // Auto-generated TOC with custom style
            const result = generateCustomStyledTOC(html, tocStyle);
            toc = result.toc;
            modifiedHtml = result.modifiedHtml;
          }
        }
        
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
          readingTime,
          tocStyle
        });
        console.log(`Processed: ${file} -> ${slug} (TOC: ${tocStyle})`);
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
        <a href="posts/${p.slug}.html" class="post-card-enhanced group">
          <div class="flex flex-col md:flex-row h-full">
            ${p.image ? `
              <div class="md:w-1/3 w-full h-48 md:h-auto flex-shrink-0 overflow-hidden">
                <img src="${p.image}" alt="${htmlEscape(p.title)}" class="post-card-image w-full h-full object-cover">
              </div>
            ` : ''}
            <div class="p-6 flex flex-col justify-between flex-grow ${p.image ? '' : 'w-full'}">
              <div>
                <div class="flex items-center justify-between mb-2">
                  <p class="post-meta text-xs text-gray-400">${htmlEscape(
                    new Date(p.date).toLocaleDateString()
                  )}</p>
                  <span class="post-meta text-xs text-gray-500">${p.readingTime} min read</span>
                </div>
                <h3 class="post-card-title text-lg font-bold mb-2 text-gray-200">${htmlEscape(p.title)}</h3>
                <p class="text-sm text-gray-300 mb-3 line-clamp-3">${htmlEscape(p.excerpt)}</p>
              </div>
              ${p.tags && p.tags.length > 0 ? `
                <div class="flex flex-wrap gap-2 mt-4">
                  ${p.tags.map(tag => `
                    <span class="tag px-2 py-1 bg-gray-700 text-teal-400 text-xs rounded-full border border-gray-600">
                      ${htmlEscape(tag)}
                    </span>
                  `).join('')}
                </div>
              ` : ''}
            </div>
          </div>
        </a>`
    )
    .join('\n        ');

  const templatePath = path.join(__dirname, 'index.html');
  let template = fs.readFileSync(templatePath, 'utf8');
  return template.replace('<!-- POST_CARDS -->', postCards);
}

function renderPostTemplate(post) {
  return `<!doctype html>
<html lang="en" class="bg-gray-900 text-gray-200">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>${htmlEscape(post.title)} - Sameer's Blog</title>
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://cdn.jsdelivr.net/npm/prismjs/themes/prism-tomorrow.min.css" rel="stylesheet" />
<script>
window.MathJax = {
  tex: { inlineMath: [['$', '$'], ['\\\\(', '\\\\)']] },
  svg: { fontCache: 'global' }
};
</script>
<link rel="stylesheet" href="../style.css">
</head>
<body class="font-mono flex flex-col min-h-screen bg-gray-900 text-gray-200">
  <header class="flex justify-between items-center px-8 py-6 border-b border-gray-700">
    <a href="../index.html" class="flex items-center hover:opacity-80 transition-opacity">
      <div class="w-10 h-10 bg-gradient-to-br from-teal-400 to-blue-500 rounded-full flex items-center justify-center text-white font-bold text-lg">
        S
      </div>
    </a>
    <nav class="flex space-x-6 text-sm">
      <a href="../index.html" class="hover:text-teal-400">Home</a>
      <a href="../about.html" class="hover:text-teal-400">About</a>
      <a href="../skills.html" class="hover:text-teal-400">Skills</a>
      <a href="../contact.html" class="hover:text-teal-400">Contact</a>
    </nav>
  </header>

  <main class="max-w-3xl mx-auto py-8 px-4 flex-grow">
    <div class="flex items-center justify-between mb-2">
      <p class="text-sm text-gray-400">${htmlEscape(
        new Date(post.date).toLocaleDateString()
      )}</p>
      <span class="text-sm text-gray-500">${post.readingTime} min read</span>
    </div>
    
    <h1 class="text-3xl font-bold mb-4">${htmlEscape(post.title)}</h1>
    
    ${post.tags && post.tags.length > 0 ? `
      <div class="flex flex-wrap gap-2 mb-6">
        ${post.tags.map(tag => `
          <span class="px-3 py-1 bg-gray-700 text-teal-400 text-sm rounded-full">
            ${htmlEscape(tag)}
          </span>
        `).join('')}
      </div>
    ` : ''}
    
    ${post.image ? `<img src="../${post.image}" alt="${htmlEscape(post.title)}" class="post-feature-img mb-6"/>` : ''}
    
    ${post.toc}
    
    <article class="prose prose-invert max-w-none">
      ${post.html}
    </article>
  </main>

  <footer class="text-center text-gray-400 text-sm py-6 border-t border-gray-700">
    <div>&copy; ${new Date().getFullYear()} ${htmlEscape(siteMeta.title)}</div>
  </footer>

<script src="https://cdn.jsdelivr.net/npm/prismjs/prism.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/prismjs/components/prism-python.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/prismjs/components/prism-javascript.min.js"></script>
<script>Prism.highlightAll();</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>

<script>
// Smooth scrolling for TOC links
document.addEventListener('DOMContentLoaded', function() {
  const tocLinks = document.querySelectorAll('.toc a[href^="#"], .toc-minimal a[href^="#"], .toc-sidebar a[href^="#"]');
  tocLinks.forEach(link => {
    link.addEventListener('click', function(e) {
      e.preventDefault();
      const targetId = this.getAttribute('href').substring(1);
      const targetElement = document.getElementById(targetId);
      if (targetElement) {
        targetElement.scrollIntoView({ 
          behavior: 'smooth',
          block: 'start'
        });
      }
    });
  });
  
  // Optional: Highlight current section in TOC
  const headings = document.querySelectorAll('h2, h3, h4, h5, h6');
  const tocLinks = document.querySelectorAll('.toc a, .toc-minimal a, .toc-sidebar a');
  
  function highlightCurrentSection() {
    let current = '';
    headings.forEach(heading => {
      const rect = heading.getBoundingClientRect();
      if (rect.top <= 100) {
        current = heading.id;
      }
    });
    
    tocLinks.forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href') === '#' + current) {
        link.classList.add('active');
      }
    });
  }
  
  window.addEventListener('scroll', highlightCurrentSection);
  highlightCurrentSection(); // Initial call
});
</script>
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
    console.log('📂 Open dist/index.html to view your blog');
  } catch (err) {
    console.error('❌ Build failed:', err);
    process.exit(1);
  }
})();