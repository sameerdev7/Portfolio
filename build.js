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
  baseUrl: '' // e.g. '/myblog' if hosting under a subpath
};

const htmlEscape = (s = '') =>
  String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

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
    // Ignore missing assets
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
        posts.push({
          title: data.title || slug,
          date: data.date || '1970-01-01',
          excerpt:
            data.excerpt ||
            (content.split('\n').find(l => l.trim()) || '').slice(0, 160),
          image: data.image || '',
          slug,
          html
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
        <a href="posts/${p.slug}.html" class="flex flex-col md:flex-row bg-gray-800 rounded-lg overflow-hidden shadow-md hover:shadow-lg hover:bg-gray-750 transition duration-200">
          ${p.image ? `<img src="${p.image}" alt="${htmlEscape(p.title)}" class="md:w-1/3 w-full object-cover">` : ''}
          <div class="p-6 flex flex-col justify-center">
            <p class="text-xs text-gray-400 mb-1">${htmlEscape(
              new Date(p.date).toLocaleDateString()
            )}</p>
            <h3 class="text-lg font-bold mb-2">${htmlEscape(p.title)}</h3>
            <p class="text-sm text-gray-300">${htmlEscape(p.excerpt)}</p>
          </div>
        </a>`
    )
    .join('\n        ');

  const templatePath = path.join(__dirname, 'index.html');
  let template = fs.readFileSync(templatePath, 'utf8');

  // Replace placeholder with cards
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
    <a href="../index.html" class="text-xl font-bold hover:text-teal-400">Sameer's Blog</a>
    <nav class="flex space-x-6 text-sm">
      <a href="../index.html" class="hover:text-teal-400">Home</a>
      <a href="../about.html" class="hover:text-teal-400">About</a>
      <a href="../skills.html" class="hover:text-teal-400">Skills</a>
    </nav>
  </header>

  <main class="max-w-3xl mx-auto py-8 px-4 flex-grow">
    <p class="text-sm text-gray-400 mb-2">${htmlEscape(
      new Date(post.date).toLocaleDateString()
    )}</p>
    <h1 class="text-3xl font-bold mb-4">${htmlEscape(post.title)}</h1>
    ${post.image ? `<img src="../${post.image}" alt="${htmlEscape(post.title)}" class="post-feature-img mb-6"/>` : ''}
    <article class="prose prose-invert max-w-none">
      ${post.html}
    </article>
  </main>

  <footer class="text-center text-gray-400 text-sm py-6 border-t border-gray-700">
    <div class="mb-2">Connect with me — 
      <a href="https://github.com/" target="_blank" class="hover:text-teal-400">GitHub</a> • 
      <a href="https://twitter.com/" target="_blank" class="hover:text-teal-400">Twitter</a> • 
      <a href="https://linkedin.com/" target="_blank" class="hover:text-teal-400">LinkedIn</a>
    </div>
    <div>&copy; ${new Date().getFullYear()} ${htmlEscape(siteMeta.title)}</div>
  </footer>

<script src="https://cdn.jsdelivr.net/npm/prismjs/prism.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/prismjs/components/prism-python.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/prismjs/components/prism-javascript.min.js"></script>
<script>Prism.highlightAll();</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
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

  const staticFiles = ['about.html', 'skills.html', 'style.css'];
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