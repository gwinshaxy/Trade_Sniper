const CACHE_NAME = 'tradesniper-v1';

self.addEventListener('install', (event) => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(clients.claim());
});

self.addEventListener('fetch', (event) => {
  // Let Streamlit handle API and WebSocket connections dynamically
  event.respondWith(
    fetch(event.request).catch(() => {
      return new Response('Offline - Reconnecting to MEXC Terminal...');
    })
  );
});