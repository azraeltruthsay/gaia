/* Light or dark, remembered per browser (the choice is applied in <head> first,
   so the page doesn't flash). Dark is the default. */
(function () {
  function current() { return document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark'; }
  function label(button) {
    const light = current() === 'light';
    button.textContent = light ? 'Dark mode' : 'Light mode';
    button.setAttribute('aria-pressed', String(light));
  }
  document.addEventListener('DOMContentLoaded', () => {
    const button = document.getElementById('theme-toggle');
    if (!button) return;
    label(button);
    button.addEventListener('click', () => {
      const next = current() === 'light' ? 'dark' : 'light';
      document.documentElement.setAttribute('data-theme', next);
      try { localStorage.setItem('gaia-theme', next); } catch (e) { /* still applies to this page */ }
      label(button);
    });
  });
})();
