#pragma TextEncoding = "UTF-8"
#pragma rtGlobals=3		// Use modern global access method and strict wave access.

Function convert_to_hdf5(filename)

    String filename   
    Variable root_id, h5_id
    HDF5CreateFile /O h5_id as filename
    HDF5CreateGroup h5_id, "/", root_id
    HDF5SaveGroup  /VAR=1 /L=7 /O /R :, root_id, "/"
    HDF5CloseGroup root_id
    HDF5CloseFile h5_id 
end


// Given a path to a folder on disk, gets all files ending in "ext" 
Function/S findFiles(path, ext[, recurse])
// By default, subfolders are searched. Turn off with recurse=0.
// 2017-05-02, joel corbin
//
    string path, ext; variable recurse
    
    if (paramIsDefault(recurse))
        recurse=1
    endif
    path = sanitizeFilePath(path)                                               // may not work in extreme cases
    
    string fileList=""
    string files=""
    string pathName = "tmpPath"
    string folders =path+";"                                                    // Remember the full path of all folders in "path" & search each for "ext" files
    string fldr
    do
        fldr = stringFromList(0,folders)
        NewPath/O/Q $pathName, fldr                                             // sets S_path=$path, and creates the symbolic path needed for indexedFile()
        PathInfo $pathName
        files = indexedFile($pathName,-1,ext)                                   // get file names
        if (strlen(files))
            files = fldr+":"+ replaceString(";", removeEnding(files), ";"+fldr+":") // add the full path (folders 'fldr') to every file in the list
            fileList = addListItem(files,fileList)
        endif
        if (recurse)
            folders += indexedDir($pathName,-1,1)                               // get full folder paths
        endif
        folders = removeFromList(fldr, folders)                                 // Remove the folder we just looked at
    while (strlen(folders))
    KillPath $pathName
    return fileList
End


static function /s sanitizeFilePath(path)
    // Avoid annoyances with escape characters when using Microsoft Windows directories.
    
    string path
    path = replaceString("\t", path, "\\t")
    path = replaceString("\r", path, "\\r")
    path = replaceString("\n", path, "\\n")

    return path
end

Function CleanUpWorkSpace()
	KillWaves/A
	KillVariables/A
	KillStrings/A
	KillDataFolder GetDataFolder(1)
end

Function convert([,i])
   variable i
	
   GetFileFolderInfo /D

   string fileList, fname, fname2
	string FILE_LOCATION="C:\\temp\\file.h5"
	
	filelist = findFiles(S_path,".pxp")
	do
		fname = stringfromlist(i,filelist)
		fname2 = ReplaceString(".pxp", fname, ".h5")
		CleanUpWorkSpace()
		LoadData/O/L=7/R fname
		Print "i=", i, ") generating", fname2
		print fname2
		convert_to_hdf5(fname2)	
		//convert_to_hdf5(FILE_LOCATION)	
		//MoveFile /I=0 /O FILE_LOCATION as fname2
		i += 1          //move to next file
    while(i<itemsinlist(filelist))
    Print "[ok] successful conversion of", S_path
End
