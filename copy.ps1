$sourcePath = "D:\comparison_8_10"  # Replace with your source path
$destinationPath = "C:\Users\matth\Desktop\PaperII\data\ballsize\"  # Replace with your destination path

# Get all subfolders in the source path
$subfolders = Get-ChildItem -Path $sourcePath -Directory

foreach ($folder in $subfolders) {
    $folderPath = $folder.FullName
    $destinationFolderPath = Join-Path -Path $destinationPath -ChildPath $folder.Name

    try {
        # Check if the folder already exists in the destination path
        if (-not (Test-Path -Path $destinationFolderPath)) {
            # Copy the folder and its contents to the destination path
            Copy-Item -Path $folderPath -Destination $destinationPath -Recurse -ErrorAction Stop
            Write-Host "Copied folder: $($folder.Name)"
        }
    } catch {
        # Print the folder name if an exception occurs
        Write-Host "Exception occurred for folder: $($folder.Name)"
    }
}