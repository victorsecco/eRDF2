$source = "C:\Users\seccolev\eRDF2\GUI\mypackages\"
$target = "C:\Users\seccolev\data_processing\src\mypackages\"

Copy-Item -Recurse -Force -Exclude '__pycache__' "$source*" "$target"
Write-Host "✅ mypackages synced from GUI to data_processing."