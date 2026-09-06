# Portal

Eksperimen computer-vision real-time untuk membuat bidang “portal” di antara dua tangan langsung dari kamera HP.

## V1

- Kamera depan/belakang via `getUserMedia()`.
- MediaPipe Hand Landmarker, maksimum 2 tangan.
- Empat sudut portal berasal dari ujung telunjuk + jempol kedua tangan.
- Smoothing landmark supaya bidang tidak terlalu jitter.
- Isi portal = salinan frame kamera yang tetap sejajar dengan scene asli, lalu difilter hanya di dalam quadrilateral.
- Filter: Thermal, Mono, Neon, Invert, Raw.
- Debug skeleton/landmark tangan.
- FPS counter.
- PWA shell untuk dipasang ke homescreen.
- GPU delegate dicoba lebih dulu, lalu fallback CPU.

## Menjalankan

Kamera browser membutuhkan **secure context**. Jalankan dari HTTPS (mis. Vercel/GitHub Pages) atau localhost. Membuka `index.html` langsung sebagai `file://` tidak cukup untuk akses kamera di browser Android.

Tidak ada backend dan tidak ada API key. Frame kamera diproses di perangkat oleh MediaPipe.

## Struktur

- `index.html` — UI.
- `app.js` — kamera, MediaPipe, geometri portal, filter, rendering.
- `styles.css` — HUD mobile fullscreen.
- `sw.js` + `manifest.webmanifest` — PWA shell.

## Catatan performa

V1 sengaja sederhana. Filter thermal masih berbasis pixel buffer CPU pada buffer yang diperkecil; tahap berikutnya adalah memindahkan filter ke WebGL shader agar lebih cepat dan menambah efek portal yang lebih kompleks.