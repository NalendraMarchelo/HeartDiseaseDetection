#!/bin/bash
# deploy.sh

echo "Merestart layanan aplikasi untuk memuat model baru..."
docker-compose up --build -d app
echo "✅ Aplikasi berhasil di-restart dengan model terbaru."
echo "melihat log dengan 'docker-compose logs -f app'"