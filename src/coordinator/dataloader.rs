use byteorder::{LittleEndian, ReadBytesExt};
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use protean::embedding_space::{Embedding, EmbeddingSpace, F32L2Space};
use suppaftp::FtpStream;

#[derive(Debug, Clone)]
pub struct DataSet<S: EmbeddingSpace> {
    pub train: Vec<S::EmbeddingData>,
    pub test: Vec<S::EmbeddingData>,
}

pub trait DataLoader<S: EmbeddingSpace> {
    fn verify_available(&self) -> bool;
    fn download(&self) -> Result<(), Box<dyn std::error::Error>>;
    fn load_data(self) -> Result<DataSet<S>, Box<dyn std::error::Error>>;
}


type Sift1MSpace = F32L2Space<128>;

pub struct Sift1MDataset {
    archive_url: &'static str,
    archive_name: &'static str,
    base_file_name: &'static str,
    query_file_name: &'static str,
    data_directory: PathBuf,
}

impl Sift1MDataset {
    pub fn new(data_dir: &str) -> Self {
        Self {
            archive_url: "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
            archive_name: "sift.tar.gz",
            // The tar.gz extracts to a sift/ subdirectory containing all files
            base_file_name: "sift/sift_base.fvecs",
            query_file_name: "sift/sift_query.fvecs",
            data_directory: PathBuf::from(data_dir),
        }
    }

    fn read_fvecs<S: EmbeddingSpace, P: AsRef<Path>>(path: P) -> io::Result<Vec<S::EmbeddingData>>
    where
        S::EmbeddingData: Embedding<Scalar = f32>,
    {
        let mut file = File::open(path)?;
        let mut data: Vec<S::EmbeddingData> = Vec::new();

        loop {
            // Read the dimension (d) of the vector as a little-endian i32
            let d = match file.read_i32::<LittleEndian>() {
                Ok(val) => val as usize,
                // Check for End of File (EOF)
                Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            };

            // Read 'd' number of float32 values
            let mut vector = Vec::with_capacity(d);
            for _ in 0..d {
                let float_val = file.read_f32::<LittleEndian>()?;
                vector.push(float_val);
            }

            data.push(S::create_embedding(vector));
        }

        Ok(data)
    }
}

impl DataLoader<Sift1MSpace> for Sift1MDataset {
    fn verify_available(&self) -> bool {
        let base_path = self.data_directory.join(self.base_file_name);
        let query_path = self.data_directory.join(self.query_file_name);

        base_path.exists() && query_path.exists()
    }


    fn download(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.verify_available() {
            tracing::info!("Dataset already available at {}. Skipping download.", self.data_directory.display());
            return Ok(());
        }

        fs::create_dir_all(&self.data_directory)?;
        let archive_path = self.data_directory.join(self.archive_name);

        // Check if archive already exists (user may have downloaded manually)
        if archive_path.exists() {
            tracing::info!("Found existing archive at {}, extracting...", archive_path.display());
        } else {
            // Download via FTP (like Python's urllib with ftp:// URLs)
            tracing::info!("Downloading dataset from {}...", self.archive_url);
            tracing::info!("   (~160MB, this may take a few minutes)");

            // Connect to FTP server
            let mut ftp = FtpStream::connect("ftp.irisa.fr:21")
                .map_err(|e| format!("FTP connection failed: {}", e))?;
            ftp.login("anonymous", "anonymous@")
                .map_err(|e| format!("FTP login failed: {}", e))?;
            ftp.transfer_type(suppaftp::types::FileType::Binary)
                .map_err(|e| format!("Failed to set binary mode: {}", e))?;

            // Stream directly to file to avoid timeout on large downloads
            let mut file = File::create(&archive_path)?;
            ftp.retr("/local/texmex/corpus/sift.tar.gz", |stream| {
                io::copy(stream, &mut file).map_err(|e| suppaftp::FtpError::ConnectionError(e))
            }).map_err(|e| format!("FTP download failed: {}", e))?;

            let _ = ftp.quit();
            tracing::info!("   -> Download complete: {}", archive_path.display());
        }

        tracing::info!("Extracting archive to {}...", self.data_directory.display());
        let file = File::open(&archive_path)?;
        let tar_gz = flate2::read::GzDecoder::new(file);
        let mut archive = tar::Archive::new(tar_gz);

        archive.unpack(&self.data_directory)?;

        fs::remove_file(&archive_path)?;
        tracing::info!("   -> Extraction complete and archive removed.");

        Ok(())
    }

    fn load_data(self) -> Result<DataSet<Sift1MSpace>, Box<dyn std::error::Error>> {
        let base_path = self.data_directory.join(self.base_file_name);
        let query_path = self.data_directory.join(self.query_file_name);

        let train = Self::read_fvecs::<Sift1MSpace, _>(&base_path)
            .map_err(|e| format!("Failed to read base vectors from {}: {}", base_path.display(), e))?;
        let test = Self::read_fvecs::<Sift1MSpace, _>(&query_path)
            .map_err(|e| format!("Failed to read query vectors from {}: {}", query_path.display(), e))?;

        Ok(DataSet { train, test })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_sift1m_dataset_paths() {
        let dataset = Sift1MDataset::new("/tmp/test_data");
        assert_eq!(dataset.data_directory, PathBuf::from("/tmp/test_data"));
        assert_eq!(dataset.base_file_name, "sift/sift_base.fvecs");
        assert_eq!(dataset.query_file_name, "sift/sift_query.fvecs");
    }

    #[test]
    fn test_verify_available_missing_files() {
        let temp_dir = TempDir::new().unwrap();
        let dataset = Sift1MDataset::new(temp_dir.path().to_str().unwrap());

        // Neither file exists
        assert!(!dataset.verify_available());
    }

    #[test]
    fn test_verify_available_partial_files() {
        let temp_dir = TempDir::new().unwrap();
        let dataset = Sift1MDataset::new(temp_dir.path().to_str().unwrap());

        // Create only base file in sift/ subdirectory
        let sift_dir = temp_dir.path().join("sift");
        fs::create_dir_all(&sift_dir).unwrap();
        let base_path = sift_dir.join("sift_base.fvecs");
        File::create(&base_path).unwrap();

        // Still should be false (query file missing)
        assert!(!dataset.verify_available());
    }

    #[test]
    fn test_verify_available_all_files() {
        let temp_dir = TempDir::new().unwrap();
        let dataset = Sift1MDataset::new(temp_dir.path().to_str().unwrap());

        // Create both files in sift/ subdirectory
        let sift_dir = temp_dir.path().join("sift");
        fs::create_dir_all(&sift_dir).unwrap();

        let base_path = sift_dir.join("sift_base.fvecs");
        File::create(&base_path).unwrap();

        let query_path = sift_dir.join("sift_query.fvecs");
        File::create(&query_path).unwrap();

        assert!(dataset.verify_available());
    }

    #[test]
    fn test_read_fvecs_format() {
        // Create a minimal fvecs file with known data
        let temp_dir = TempDir::new().unwrap();
        let fvecs_path = temp_dir.path().join("test.fvecs");

        let mut file = File::create(&fvecs_path).unwrap();

        // Write 2 vectors of dimension 4
        // Vector 1: [1.0, 2.0, 3.0, 4.0]
        file.write_all(&4i32.to_le_bytes()).unwrap(); // dimension
        file.write_all(&1.0f32.to_le_bytes()).unwrap();
        file.write_all(&2.0f32.to_le_bytes()).unwrap();
        file.write_all(&3.0f32.to_le_bytes()).unwrap();
        file.write_all(&4.0f32.to_le_bytes()).unwrap();

        // Vector 2: [5.0, 6.0, 7.0, 8.0]
        file.write_all(&4i32.to_le_bytes()).unwrap(); // dimension
        file.write_all(&5.0f32.to_le_bytes()).unwrap();
        file.write_all(&6.0f32.to_le_bytes()).unwrap();
        file.write_all(&7.0f32.to_le_bytes()).unwrap();
        file.write_all(&8.0f32.to_le_bytes()).unwrap();

        drop(file);

        // Read using the function with F32L2Space<4>
        type TestSpace = F32L2Space<4>;
        let vectors = Sift1MDataset::read_fvecs::<TestSpace, _>(&fvecs_path).unwrap();

        assert_eq!(vectors.len(), 2);
    }

    #[test]
    fn test_read_fvecs_empty_file() {
        let temp_dir = TempDir::new().unwrap();
        let fvecs_path = temp_dir.path().join("empty.fvecs");
        File::create(&fvecs_path).unwrap();

        type TestSpace = F32L2Space<4>;
        let vectors = Sift1MDataset::read_fvecs::<TestSpace, _>(&fvecs_path).unwrap();
        assert!(vectors.is_empty());
    }

    #[test]
    fn test_dataset_struct() {
        type TestSpace = F32L2Space<4>;
        let dataset: DataSet<TestSpace> = DataSet {
            train: vec![],
            test: vec![],
        };
        assert!(dataset.train.is_empty());
        assert!(dataset.test.is_empty());
    }
}