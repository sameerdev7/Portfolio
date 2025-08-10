# Minimal Markdown Blog

A fast, minimal static blog generator that transforms Markdown posts into a beautiful dark-themed website. Perfect for developers, researchers, and writers who want a simple yet elegant blogging solution.

## ✨ Features

- **📝 Markdown-powered**: Write posts in pure Markdown with YAML frontmatter
- **🎨 Beautiful dark theme**: Easy-on-the-eyes minimal design with Tailwind CSS
- **⚡ Lightning fast**: Static HTML generation for optimal performance
- **🖼️ Featured images**: Support for post thumbnails and hero images
- **🔍 Syntax highlighting**: Code blocks with Prism.js highlighting
- **🧮 Math support**: LaTeX math rendering via MathJax
- **📱 Responsive**: Looks great on desktop, tablet, and mobile
- **🚀 Deploy anywhere**: Works on GitHub Pages, Netlify, Vercel, or any static host

## 🚀 Quick Start

### Prerequisites
- Node.js (v14 or higher)
- npm or yarn

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd minimal-md-blog
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Build the site**
   ```bash
   npm run build
   ```

4. **View your blog**
   ```bash
   # Open dist/index.html in your browser
   # Or serve with a local server:
   npx serve dist
   ```

## 📝 Writing Posts

### Create a New Post

1. Create a new `.md` file in the `posts/` directory
2. Add frontmatter at the top of your file:
   ```yaml
   ---
   title: "Your Amazing Post Title"
   date: "2025-08-10"
   excerpt: "A compelling excerpt that appears on the homepage"
   image: "assets/your-image.jpg"  # optional
   slug: "your-post-url"           # optional, defaults to filename
   ---
   ```

3. Write your content in Markdown below the frontmatter
4. Run `npm run build` to regenerate your site

### Example Post

```markdown
---
title: "Getting Started with Machine Learning"
date: "2025-08-10"
excerpt: "A beginner's guide to understanding ML fundamentals and getting started with your first project."
image: "assets/ml-intro.jpg"
slug: "ml-getting-started"
---

# Welcome to Machine Learning!

Machine learning is revolutionizing how we solve problems...

## Code Example

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Simple linear regression
model = LinearRegression()
model.fit(X_train, y_train)
```

## Math Support

Inline math: $E = mc^2$

Block math:
$$
\int_0^1 x^2 \, dx = \frac{1}{3}
$$
```

## 🛠️ Customization

### Changing the Theme

Edit `style.css` to customize colors, fonts, and layout:

```css
:root {
  --bg: #0b1220;      /* Background */
  --panel: #0f1724;   /* Cards/panels */
  --muted: #9aa8b3;   /* Muted text */
  --text: #dbeafe;    /* Main text */
  --accent: #14b8a6;  /* Accent color */
}
```

### Updating Site Information

Edit the `siteMeta` object in `build.js`:

```javascript
const siteMeta = {
  title: "Your Blog Name",
  author: 'Your Name',
  baseUrl: '' // e.g., '/myblog' for GitHub Pages
};
```

### Adding Pages

1. Create an HTML file in the root directory (e.g., `contact.html`)
2. Follow the same structure as `about.html` or `skills.html`
3. Add navigation links in your templates
4. The build script will automatically copy it to `dist/`

## 📁 Project Structure

```
your-blog/
├── 📄 index.html          # Homepage template
├── 📄 about.html          # About page
├── 📄 skills.html         # Skills page
├── 🎨 style.css           # Custom styles
├── ⚙️ build.js            # Build script
├── 📦 package.json        # Dependencies
├── 📝 README.md           # This file
├── 🚫 .gitignore         # Git ignore rules
├── 📚 posts/             # Your blog posts
│   ├── welcome.md
│   └── start.md
├── 🖼️ assets/            # Images and media
│   └── oppie.jpg
└── 🏗️ dist/             # Generated site (don't edit!)
    ├── index.html
    ├── posts/
    ├── assets/
    └── ...
```

## 🔧 Available Scripts

```bash
# Build the static site
npm run build

# Clean the dist directory
npm run clean

# Build and show completion message
npm run dev
```

## 🎯 Frontmatter Reference

| Field     | Required | Description                              | Example                    |
|-----------|----------|------------------------------------------|----------------------------|
| `title`   | ✅       | Post title (appears in header and meta) | `"My Amazing Post"`        |
| `date`    | ✅       | Publication date (YYYY-MM-DD)           | `"2025-08-10"`            |
| `excerpt` | ❌       | Short description for cards             | `"This post explains..."` |
| `image`   | ❌       | Featured image path                      | `"assets/hero.jpg"`       |
| `slug`    | ❌       | URL slug (defaults to filename)         | `"my-amazing-post"`       |

## 📄 Markdown Support

### Text Formatting
- **Bold**: `**bold text**`
- *Italic*: `*italic text*`
- `Code`: `\`inline code\``
- ~~Strikethrough~~: `~~strikethrough~~`

### Code Blocks
````markdown
```python
def hello_world():
    print("Hello, World!")
```
````

### Math (LaTeX)
- Inline: `$E = mc^2$`
- Block: `$$\int_0^1 x^2 \, dx = \frac{1}{3}$$`

### Links and Images
- Links: `[text](url)`
- Images: `![alt](path)`

## 🚀 Deployment

### GitHub Pages
1. Push your code to GitHub
2. Go to Settings → Pages
3. Set source to GitHub Actions
4. Create `.github/workflows/deploy.yml`:
   ```yaml
   name: Deploy to GitHub Pages
   on:
     push:
       branches: [ main ]
   jobs:
     build-and-deploy:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v2
         - uses: actions/setup-node@v2
           with:
             node-version: '16'
         - run: npm install
         - run: npm run build
         - uses: peaceiris/actions-gh-pages@v3
           with:
             github_token: ${{ secrets.GITHUB_TOKEN }}
             publish_dir: ./dist
   ```

### Netlify
1. Connect your GitHub repo
2. Set build command: `npm run build`
3. Set publish directory: `dist`
4. Deploy!

### Vercel
1. Import your GitHub repo
2. Framework preset: "Other"
3. Build command: `npm run build`
4. Output directory: `dist`

## 🐛 Troubleshooting

### Posts not showing up?
- Check that your `.md` file is in the `posts/` directory
- Ensure your frontmatter is valid YAML
- Run `npm run build` and check console output for errors
- Verify the `<!-- POST_CARDS -->` placeholder exists in `index.html`

### Images not loading?
- Place images in the `assets/` directory
- Use relative paths: `"assets/image.jpg"`
- Check that the file extension matches exactly

### Math not rendering?
- Ensure MathJax script is loaded in your templates
- Use proper LaTeX syntax: `$inline$` or `$$block$$`

### Build failing?
- Check that all dependencies are installed: `npm install`
- Verify Node.js version: `node --version` (should be 14+)
- Look for syntax errors in your Markdown files

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test them
4. Commit your changes: `git commit -m 'Add feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Built with [Marked](https://marked.js.org/) for Markdown parsing
- Styled with [Tailwind CSS](https://tailwindcss.com/)
- Code highlighting by [Prism.js](https://prismjs.com/)
- Math rendering by [MathJax](https://www.mathjax.org/)
- Inspired by the need for simple, fast blogging solutions

---

**Happy blogging!** 🎉

If you have questions or need help, please [open an issue](https://github.com/your-username/your-repo/issues) on GitHub.