<?php
ini_set("display_errors", "On");
ini_set("memory_limit", "10G");

use PHP2xAI\Datasets\Batches\FileBatch;
use PHP2xAI\Utility\Images\Vectorizer;

include("../vendor/autoload.php");

mt_srand(42);

$trainingSplit = 0.8;
$numberOfElements = 28 * 28; // 784
$numberOfElementsPerBatch = 120; // 80 batches for ~48k samples

$imagesPath = __DIR__."/images";
$trainingPath = __DIR__."/DataLabelInt/Training";
$testPath = __DIR__."/DataLabelInt/Test";

prepareDataPath($trainingPath);
prepareDataPath($testPath);

$trainingFiles = [];
$testFiles = [];

for ($digit = 0; $digit <= 9; $digit++)
{
	$digitPath = $imagesPath."/".$digit;
	
	if (!@is_dir($digitPath))
		continue;
	
	$digitFiles = glob($digitPath."/*.png") ?: [];
	
	if (count($digitFiles) === 0)
		continue;
	
	shuffle($digitFiles);
	
	$trainingCount = (int)round(count($digitFiles) * $trainingSplit);
	
	$trainingSlice = array_slice($digitFiles, 0, $trainingCount);
	$testSlice = array_slice($digitFiles, $trainingCount);
	
	foreach ($trainingSlice as $file)
	{
		$trainingFiles[] = ["digit" => $digit, "path" => $file];
	}
	
	foreach ($testSlice as $file)
	{
		$testFiles[] = ["digit" => $digit, "path" => $file];
	}
}

shuffle($trainingFiles);
shuffle($testFiles);

writeBatches($trainingFiles, $trainingPath."/train.txt");
writeBatches($testFiles, $trainingPath."/test.txt");

/**
 * Remove existing batches and ensure folder exists.
 */
function prepareDataPath(string $path)
{
	if (!@is_dir($path))
	{
		@mkdir($path, 0777, true);
		return;
	}
	
	$iterator = new \FilesystemIterator($path);
	
	foreach ($iterator as $item)
	{
		if ($item->isDir())
		{
			$files = glob($item->getPathname()."/*") ?: [];
			
			foreach ($files as $file)
				@unlink($file);
			
			@rmdir($item->getPathname());
		}
	}
}

/**
 * Write image vectors/labels to batched folders.
 *
 * @return array [totalSamples, batchesCreated]
 */
function writeBatches(array $files, string $targetPath)
{
	$total = count($files);
	// $batchIndex = 0;
	// $batch = new FileBatch($targetPath, $batchIndex);
	
	@unlink($targetPath);
	
	if (!$fp = fopen($targetPath, 'w+')) {
		echo "Cannot open file ($filename)";
		exit;
	};
	
	foreach ($files as $i => $fileInfo)
	{
		$vector = new Vectorizer($fileInfo["path"]);
		$x = $vector->toArray(true);
		// print_r($x);

// 		$y = array_fill(0, 1, 0);
// 		$y[(int)$fileInfo["digit"]] = 1;
// 		
		$y = $fileInfo["digit"];
		
		$row = implode(" ",$x)."|".$y."\n";
		
		// Write $somecontent to our opened file.
		if (fwrite($fp, $row) === FALSE) {
			echo "Cannot write to file ($filename)";
			exit;
		}

		
		
// 		$batch->append($x, $y);
// 		
// 		$isLastElement = $i === ($total - 1);
// 		
// 		if ($batch->n === $batchSize || $isLastElement)
// 		{
// 			$batch->write();
// 			
// 			unset($batch);
// 			
// 			if (!$isLastElement)
// 			{
// 				$batchIndex++;
// 				$batch = new FileBatch($targetPath, $batchIndex);
// 			}
// 		}
	}
	
	fclose($fp);
}
