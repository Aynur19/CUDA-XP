extern "C" void cuVectorAdd_RAPI(const int blockSize, const int numBlocks, const int numItems);
//extern "C" void cuVectorAdd_DAPI(const int blockSize, const int numBlocks, const int numItems);

int main(int argc, char* argv[])
{
	const unsigned int blockSize = 512;
	const unsigned int numBlocks = 3;
	const unsigned int numItems = numBlocks * blockSize;

	cuVectorAdd_RAPI(blockSize, numBlocks, numItems);
	//cuVectorAdd_DAPI(blockSize, numBlocks, numItems);
}