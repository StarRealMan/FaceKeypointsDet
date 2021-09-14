#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<vector>

#define BLOCK_SIZE 16
#define WIDTH 96
#define HEIGHT 96

__global__ void Gend(float* Mapd, float x_pos, float y_pos);
void Gen(float* Map, float x_pos, float y_pos);

int main()
{
    int imgid = -1;

    int size = WIDTH * HEIGHT;
    int x_pos, y_pos;

    std::ifstream fin("../data/training.csv");
	std::string line;

    std::ofstream outFile;
    outFile.open("../data/gtmap.txt");

	while(getline(fin, line))
	{
        if(imgid != -1)
        {
            int num = 0;
            // std::cout <<"原始字符串："<< line << std::endl;
            std::istringstream sin(line);
            std::vector<std::string> fields;
            std::string field;
    
            while(getline(sin, field, ','))
            {
                fields.push_back(field);
                if(num == 29)
                {
                    break;
                }
                else
                {
                    num++;
                }
            }
    
            for(int kpid = 0; kpid < 15; kpid++)
            {
                if(!fields[2 * kpid].empty())
                {
                    x_pos = std::stof(fields[2 * kpid]);
                    y_pos = std::stof(fields[2 * kpid + 1]);
    
                    float* Map;
                    Map = (float*)malloc(size * sizeof(float));
                    Gen(Map, x_pos, y_pos);
                    
                    for(int i = 0; i < size; i++)
                    {
                        outFile << Map[i];
                        outFile << " ";
                    }
                }

                outFile << "\n";
            }
        }

        imgid++;
	}

    return 0;
}

__global__ void Gend(float* Mapd, float x_pos, float y_pos)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int map_x = blockDim.x * bx + tx;
    int map_y = blockDim.y * by + ty;

    int data_idx = gridDim.x * blockDim.x * map_y + map_x;

    float dist_x = (float)map_x - x_pos;
    float dist_y = (float)map_y - y_pos;
    
    float dist = sqrt(dist_x * dist_x + dist_y * dist_y);

    if(dist < 1)
    {
        Mapd[data_idx] = 1;
    }
    else if(dist < 2)
    {
        Mapd[data_idx] = 0.8;
    }
    else
    {
        Mapd[data_idx] = 1/dist;
    }

    __syncthreads();
}

void Gen(float* Map, float x_pos, float y_pos)
{
    float* Mapd;
    int size = WIDTH * HEIGHT;
    cudaMalloc((void**)&Mapd, size * sizeof(float));

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(WIDTH/dimBlock.x, HEIGHT/dimBlock.y);

    Gend<<<dimGrid, dimBlock>>>(Mapd, x_pos, y_pos);

    cudaMemcpy(Map, Mapd, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(Mapd);
}


// compile : nvcc -o gen_gthmap gen_gthmap.cu
// run     : ./gen_gthmap
