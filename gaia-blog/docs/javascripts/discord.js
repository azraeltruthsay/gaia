// Discord discussion link — injected at the bottom of blog posts
document.addEventListener("DOMContentLoaded", function () {
  const article = document.querySelector("article.md-content__inner");
  if (!article) return;

  // Skip pages that opt out
  if (document.querySelector('meta[name="comments"][content="false"]')) return;

  const container = document.createElement("div");
  container.className = "discord-cta";
  container.innerHTML = `
    <hr>
    <p>
      <strong>Want to discuss this post?</strong><br>
      Join the conversation on
      <a href="https://discord.gg/nrQ7H5fG" target="_blank" rel="noopener">
        our Discord server
      </a>
      — where GAIA is a live participant.
    </p>
  `;
  article.appendChild(container);
});
