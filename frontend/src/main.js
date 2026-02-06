import { initEeg } from './eeg.js';
import { initChat } from './chat.js';

const navBtns = document.querySelectorAll('.nav-btn');
const pages = document.querySelectorAll('.page');

navBtns.forEach((btn) => {
  btn.addEventListener('click', () => {
    const page = btn.dataset.page;
    navBtns.forEach((b) => b.classList.toggle('active', b.dataset.page === page));
    pages.forEach((p) => p.classList.toggle('active', p.id === `page-${page}`));
  });
});

initEeg();
initChat();
