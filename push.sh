git config --global user.email "shy3765@163.com"
git config --global user.name "Hoyant_Su"
git config --global credential.helper store
cd /inspire/hdd/project/continuinglearning/suhaoyang-240107100018/suhaoyang-240107100018/storage/RA-CMFormer

read -p "Input your GitHub token: " TOKEN
git remote set-url origin "https://${TOKEN}@github.com/Hoyant-Su/RA-CMFormer.git"

git add .
git commit -m "RA-CMFormer released."
git push --set-upstream origin main
