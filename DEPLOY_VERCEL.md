# Vercel deployment (Flask + Neon + Cloudflare R2)

Ye project Vercel ke zero-config Flask runtime par deploy hota hai. Vercel
filesystem read-only hai (sirf `/tmp` writable), isliye permanent data ke liye
**Neon PostgreSQL** aur photos ke liye **Cloudflare R2** zaroori hain.

## 1. Neon PostgreSQL

1. [Neon](https://console.neon.tech) par project banayein. Region apne users ke
   sabse qareeb chunein (India ke liye `AWS ap-south-1` / Mumbai).
2. Dashboard -> **Connect** -> **Pooled connection** copy karein. Host me
   `-pooler` hona chahiye.
3. Connection string ke end me `?sslmode=require` hona chahiye, e.g.:

   ```text
   postgresql://user:password@ep-xxxx-pooler.ap-south-1.aws.neon.tech/neondb?sslmode=require
   ```

4. App `postgres://` aur `postgresql://` dono ko automatically
   `postgresql+psycopg://` me convert kar deta hai. Kuch manually badalne ki
   zaroorat nahi.

Schema pehli cold start par `db.create_all()` se ban jata hai aur default admin
seed hota hai.

## 2. Cloudflare R2 (recommended)

Vercel par local photo storage permanent nahi hai. R2 set na karein to photos
`/tmp/uploads` me jayengi aur naye deployment ke baad gayab ho jayengi.

1. Cloudflare dashboard -> **R2** -> bucket banayein.
2. **Manage R2 API Tokens** -> **Create API token** -> Object Read & Write.
3. Bucket ko public karein (custom domain ya `r2.dev` URL) aur wo URL
   `R2_PUBLIC_URL` me dalein. Public URL optional hai: blank chhodne par app
   1-hour presigned URLs generate karta hai, lekin public URL caching ke liye
   behtar hai.
4. Ye values note karein: Account ID, Access Key ID, Secret Access Key, bucket
   name, public URL.

## 3. Vercel project

1. Code GitHub par push karein.
2. Vercel -> **Add New** -> **Project** -> repository import karein.
3. Framework preset **Flask** auto-detect ho jayega. Build/install settings
   `vercel.json` se aati hain, dashboard me override na karein.
4. **Settings -> Environment Variables** me (Production + Preview) add karein:

   | Variable | Required | Notes |
   | --- | --- | --- |
   | `DATABASE_URL` | Haan | Neon **pooled** connection string, `sslmode=require` ke saath |
   | `SECRET_KEY` | Haan | Lamba random string; iske bina har instance par login session reset hoga |
   | `R2_ACCOUNT_ID` | Recommended | Photos permanent rakhne ke liye |
   | `R2_ACCESS_KEY_ID` | Recommended | |
   | `R2_SECRET_ACCESS_KEY` | Recommended | |
   | `R2_BUCKET` | Recommended | |
   | `R2_PUBLIC_URL` | Optional | Blank = presigned URLs |
   | `ADMIN_USERNAME` | Optional | Default `admin` |
   | `ADMIN_PASSWORD` | Optional | **Pehle deploy se pehle badal dein**; default `admin123` |
   | `ADMIN_EMAIL` | Optional | Default `admin@example.com` |

   `ADMIN_*` sirf tab use hote hain jab database me koi admin na ho. Admin
   password baad me env se change karne se existing account update nahi hota.

5. **Deploy** dabayein. Pehla deployment face-recognition models (~100MB)
   download/build karne ki wajah se slow ho sakta hai.

CLI se deploy karna ho to:

```bash
npm i -g vercel
vercel login
vercel link
vercel env add DATABASE_URL production
# ...baaki env vars bhi isi tarah add karein
vercel --prod
```

## 4. Region

Vercel function aur Neon database ka region paas hona chahiye, warna har query
me latency milegi. India ke liye Vercel project region `Mumbai (bom1)` aur Neon
region `ap-south-1` chunein. Vercel me region project creation ke waqt ya
**Settings -> Functions -> Region** se set hota hai.

## 5. Local testing

```bash
./install.sh
cp .env.example .env   # DATABASE_URL + R2 values bharein
python app.py
```

Vercel jaisa environment test karne ke liye:

```bash
vercel env pull .env.local
cp .env.local .env     # python-decouple .env padhta hai
vercel dev
```

## 6. Vercel limits / behaviour

- **Request body limit 4.5MB**: app Vercel par upload ko 4MB tak restrict karta
  hai. Isse badi photo par friendly error milega. Employees ki photo compress
  karke upload karein.
- **`/tmp` ephemeral hai**: SQLite fallback aur local photos redeploy ke baad
  delete ho jate hain. Isliye Neon + R2 use karein. Bina `DATABASE_URL` ke app
  chalega, par banner warning dikhayega aur data permanent nahi hoga.
- **Function duration**: `maxDuration` 60 seconds set hai (face encoding ke liye).
- **Bundle size**: standard limit 500MB. Isi liye `requirements.txt` me
  `opencv` nahi hai (app use nahi karta) aur `face-recognition` alag se
  `--no-deps` install hota hai.
- **Cold start**: pehli request thodi slow ho sakti hai.
- **Static files**: `public/static/**` Vercel CDN se serve hota hai. Private
  photos wahan na rakhein; wo R2 ya `/tmp` me jati hain.

## 7. dlib / face-recognition build

`dlib` Linux par source se compile hota hai aur Vercel build me fail/hang hota
hai. Isliye:

- `requirements.txt` me `dlib-bin` hai (prebuilt `dlib` module).
- `face-recognition` requirements me nahi hai.
- `vercel.json` ka `installCommand` use karta hai:

  ```bash
  pip install -r requirements.txt && pip install --no-deps face-recognition==1.3.0 face-recognition-models==0.3.0
  ```

Agar dependencies badlein to `vercel.json` ka install command bhi update karein,
warna build purane command se chalega.

## 8. Common problems

| Problem | Fix |
| --- | --- |
| Build par `dlib` compile error | Check karein deployment Vercel project ke root se ho raha hai aur `vercel.json` use ho raha hai |
| `connection timed out` / `too many connections` | Neon ka **pooled** (`-pooler`) connection string use karein |
| Bar bar login logout ho raha hai | `SECRET_KEY` env var set karein |
| Photos redeploy ke baad gayab | R2 env vars set karein (`R2_*`) |
| `413` / file too large | Photo 4MB se chhoti karein |
| Tables missing / DB error | `DATABASE_URL` check karein aur deployment logs dekhein |
