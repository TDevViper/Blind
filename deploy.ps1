# ==============================================================================
# BLIND Assistive Navigation Platform - Manual Deployment & Git Push Script
# ==============================================================================

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  BLIND AI: Git Push & Deployment Automation  " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Step 1: Stage all changes
Write-Host "`n[1/4] Staging modified files for git commit..." -ForegroundColor Yellow
git add .

# Step 2: Prompt for commit message or use default
$commitMsg = Read-Host "`nEnter commit message (Press Enter for default: 'feat: safety case remediation and production deployment configs')"
if ([string]::IsNullOrWhiteSpace($commitMsg)) {
    $commitMsg = "feat: safety case remediation and production deployment configs"
}

Write-Host "`n[2/4] Committing changes with message: '$commitMsg'..." -ForegroundColor Yellow
git commit -m "$commitMsg"

# Step 3: Push to remote GitHub repository
Write-Host "`n[3/4] Pushing commits to remote origin (main)..." -ForegroundColor Yellow
git push origin main

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[ERROR] Git push failed! Please check your credentials or network connection." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n[4/4] Git push completed successfully!" -ForegroundColor Green
Write-Host "`n==================================================" -ForegroundColor Cyan
Write-Host "           MANUAL DEPLOYMENT INSTRUCTIONS          " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "`n1. RENDER (Python Flask + SocketIO Backend):" -ForegroundColor White
Write-Host "   - Your repository contains 'render.yaml' and 'requirements_prod.txt'." -ForegroundColor Gray
Write-Host "   - If connected via Render Blueprint / GitHub integration, Render will auto-deploy." -ForegroundColor Gray
Write-Host "   - To deploy manually: Go to https://dashboard.render.com -> Select 'blind-ai-backend' -> Click 'Manual Deploy' -> 'Deploy latest commit'." -ForegroundColor Gray
Write-Host "   - Once deployed, copy your Render backend URL (e.g., https://blind-ai-backend.onrender.com)." -ForegroundColor Green

Write-Host "`n2. VERCEL (Next.js Cyber-Cockpit Frontend):" -ForegroundColor White
Write-Host "   - Your repository contains 'vercel.json' configured for the 'frontend/' directory." -ForegroundColor Gray
Write-Host "   - If connected via Vercel GitHub integration, Vercel will auto-deploy." -ForegroundColor Gray
Write-Host "   - To deploy manually via Vercel CLI: Run 'npx vercel --prod' inside the 'frontend/' folder." -ForegroundColor Gray
Write-Host "   - IMPORTANT: In Vercel Project Settings -> Environment Variables, set:" -ForegroundColor Yellow
Write-Host "     NEXT_PUBLIC_BACKEND_URL = <Your-Render-Backend-URL>" -ForegroundColor Cyan
Write-Host "`n==================================================`n" -ForegroundColor Cyan
