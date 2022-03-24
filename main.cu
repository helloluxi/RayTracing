
#include "raytracing.cu"
#include <iostream>
#include <jsoncpp/json/json.h>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

using namespace std;

const float gammaCorrector = 2.2f;
const float invGamma = 1.0f / gammaCorrector;

void LoadTexture(Texture*& tex, const char* path);
void LoadObjModel(TriangleMesh* mesh, string& path);
void readFloat3(Json::Value& json, string key, float3& v);
void readFloat(Json::Value& json, string key, float& f);
void readInt(Json::Value& json, string key, int& i);
void readString(Json::Value& json, string key, string& s);
#define parseInt(name) int name; readInt(root, #name, name);
#define parseFloat(name) float name; readFloat(root, #name, name);
#define parseFloat3(name) float3 name; readFloat3(root, #name, name);
#define parseString(name) string name; readString(root, #name, name);
#define invalidParameter(name) catch(...){ cerr << "Invalid parameter \"" << name << "\" at line " << __LINE__ << endl; exit(1); }

int main(int argc, char** argv)
{
    auto awakeTime = clock();

    // read config
    Json::Value root;
    {
        string inputPath = argc == 1 ? "config.json" : argv[1];
        ifstream f;
        f.open(inputPath);
        if(!f.is_open()) { 
            cerr << "Cannot find file: " << inputPath << endl; 
            exit(1); 
        }
        Json::Reader reader;
        if(!reader.parse(f, root))
        {
            cerr << "Parse Error: " << inputPath << endl;
            exit(1);
        }
    }

    parseInt(myThreadNum)
    parseInt(screenWidth)
    parseInt(screenHeight)
    parseInt(nSample)
    parseInt(maxDepth)
    parseFloat(verticalFOV)
    parseFloat3(cameraPos)
    parseFloat3(cameraFront)
    parseFloat3(cameraUp)
    parseFloat3(cameraRight)
    parseString(outputPath)

    int pixelNum = screenWidth * screenHeight;
    int myGridNum = (pixelNum + myThreadNum - 1) / myThreadNum;
    

    // init random seed
    curandState* rd;
    checkCudaErrors(cudaMalloc((void**)&rd, pixelNum * sizeof(curandState)));
    runInit(rd, myGridNum, myThreadNum);
    checkCudaErrors(cudaDeviceSynchronize());


    // init scene
    Scene scene;
    try{
        auto node = root["materials"];
        if(node.isNull()){
            scene.materialCount = 0;
        }
        else{
            scene.materialCount = node.size();
            checkCudaErrors(cudaMallocManaged(&scene.materials, sizeof(Shader) * scene.materialCount));
            for (int i = 0; i < scene.materialCount; i++)
            {
                readFloat3(node[i], "albedo", scene.materials[i].albedo);
                readFloat(node[i], "metallic", scene.materials[i].metallic);
                readFloat(node[i], "roughness", scene.materials[i].roughness);
            }
        }
    } invalidParameter("materials")
    try{
        auto node = root["planes"];
        if(node.isNull()){
            scene.planeCount = 0;
        }
        else{
            scene.planeCount = node.size();
            checkCudaErrors(cudaMallocManaged(&scene.planes, sizeof(Plane) * scene.planeCount));
            for (int i = 0; i < scene.planeCount; i++)
            {
                readFloat3(node[i], "point", scene.planes[i].point);
                readFloat3(node[i], "normal", scene.planes[i].normal);
                readInt(node[i], "materialId", scene.planes[i].materialId);
            }
        }
    } invalidParameter("planes")
    try{
        auto node = root["spheres"];
        if(node.isNull()){
            scene.sphereCount = 0;
        }
        else{
            scene.sphereCount = node.size();
            checkCudaErrors(cudaMallocManaged(&scene.spheres, sizeof(Sphere) * scene.sphereCount));
            for (int i = 0; i < scene.sphereCount; i++)
            {
                readFloat3(node[i], "center", scene.spheres[i].center);
                readFloat(node[i], "r", scene.spheres[i].r);
                readInt(node[i], "materialId", scene.spheres[i].materialId);
            }
        }
    } invalidParameter("spheres")
    try{
        auto node = root["meshes"];
        if(node.isNull()){
            scene.meshCount = 0;
        }
        else{
            scene.meshCount = node.size();
            checkCudaErrors(cudaMallocManaged(&scene.meshes, sizeof(TriangleMesh) * scene.meshCount));
            for (int i = 0; i < scene.meshCount; i++)
            {
                string objPath;
                readString(node[i], "path", objPath);
                LoadObjModel(&scene.meshes[i], objPath);
                readInt(node[i], "materialId", scene.meshes[i].materialId);
            }
        }
    } invalidParameter("meshes")
    try{
        auto node = root["areaLight"];
        if(!node.isNull()){
            scene.useLightSampling = true;
            checkCudaErrors(cudaMallocManaged(&scene.light, sizeof(AreaLight)));
            readFloat3(node, "center", scene.light->center);
            readFloat3(node, "intensity", scene.light->intensity);
            readFloat3(node, "normal", scene.light->normal);
            readFloat3(node, "right", scene.light->right);
            readFloat3(node, "up", scene.light->up);
            readFloat(node, "size", scene.light->size);
        }
    } invalidParameter("areaLight")


    // Malloc result image
    float3* result;
    checkCudaErrors(cudaMallocManaged(&result, sizeof(float3) * pixelNum));


    // Ray Tracing
    auto startTime = clock();
    printf("Initializing Time = %.3fs\n", (float)(startTime - awakeTime) / CLOCKS_PER_SEC);
    runKernel(result, rd, scene,
        cameraPos, cameraFront, cameraUp, cameraRight,
        screenWidth, screenHeight, nSample, maxDepth, verticalFOV / screenHeight,
        myGridNum, myThreadNum);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("Ray Tracing Time = %.3fs\n", (float)(clock() - startTime) / CLOCKS_PER_SEC);


    // Write image to file
    stbi_uc output[pixelNum * 3];
    for (size_t i = 0; i < pixelNum; i++)
    {
        output[i * 3] = (stbi_uc)(powf(clamp(result[i].x, 0.0f, 1.0f), invGamma) * 255.f);
        output[i * 3 + 1] = (stbi_uc)(powf(clamp(result[i].y, 0.0f, 1.0f), invGamma) * 255.f);
        output[i * 3 + 2] = (stbi_uc)(powf(clamp(result[i].z, 0.0f, 1.0f), invGamma) * 255.f);
    }
    stbi_write_png(outputPath.c_str(), screenWidth, screenHeight, 3, output, screenWidth * 3);


    // Free
    checkCudaErrors(cudaDeviceReset());
    return 0;
}

void LoadTexture(Texture*& tex, const char* path) {
    checkCudaErrors(cudaMallocManaged(&tex, sizeof(Texture)));
    auto img = stbi_load(path, &tex->size[0], &tex->size[1], &tex->size[2], 3);
    if (img == NULL) {
        printf("Image %s not found!", path);
        exit(1);
    }
    checkCudaErrors(cudaMallocManaged(&tex->texture, sizeof(float3) * tex->size[0] * tex->size[1]));
    for (size_t i = 0; i < tex->size[0] * tex->size[1]; i++)
    {
        tex->texture[i] = make_float3(powf(img[i * 3] / 255.f, gammaCorrector),
            powf(img[i * 3 + 1] / 255.f, gammaCorrector), powf(img[i * 3 + 2] / 255.f, gammaCorrector));
    }
    stbi_image_free(img);
}
void readInt(Json::Value& json, string key, int& i){
    try{
        i = json[key].asInt();
    }catch(...){
        cerr << "Invalid parameter \"" << key << "\"." << endl;
        exit(1);
    }
}
void readFloat(Json::Value& json, string key, float& f){
    try{
        f = json[key].asFloat();
    }catch(...){
        cerr << "Invalid parameter \"" << key << "\"." << endl;
        exit(1);
    }
}
void readFloat3(Json::Value& json, string key, float3& v){
    try{
        v.x = json[key][0].asFloat();
        v.y = json[key][1].asFloat();
        v.z = json[key][2].asFloat();
    }catch(...){
        cerr << "Invalid parameter \"" << key << "\"." << endl;
        exit(1);
    }
}
void readString(Json::Value& json, string key, string& s){
    try{
        s = json[key].asString();
    }catch(...){
        cerr << "Invalid parameter \"" << key << "\"." << endl;
        exit(1);
    }
}
void LoadObjModel(TriangleMesh* mesh, string& path){
    vector<float3> vertices, normals;
    vector<int> triangles, triangleNormals;

    ifstream f;
    f.open(path);
    if(!f.is_open()) { 
        cerr << "Cannot find file: " << path << endl; 
        exit(1); 
    }

    try{
        char c;
        while(!f.eof()){
            f >> c;
            if(c == 'v'){
                if(f.get() == 'n'){
                    float3 v;
                    f >> c >> v.x >> v.y >> v.z;
                    normals.emplace_back(v);
                }
                else{
                    float3 v;
                    f >> v.x >> v.y >> v.z;
                    vertices.emplace_back(v);
                }
            }
            else if(c == 'f'){
                int t, n;
                f >> t >> c >> c >> n;
                triangles.push_back(t - 1);
                triangleNormals.push_back(n - 1);
                f >> t >> c >> c >> n;
                triangles.push_back(t - 1);
                triangleNormals.push_back(n - 1);
                f >> t >> c >> c >> n;
                triangles.push_back(t - 1);
                triangleNormals.push_back(n - 1);
            }
            else{
                string s;
                getline(f, s);
                // cout << "Ignore line: " << s << endl;
            }
        }
    }
    catch(...){
        cerr << "Cannot parse file: " << path << endl;
        exit(1);
    }
        
    mesh->vertexCount = vertices.size();
    mesh->triangleCount = triangles.size() / 3;
    checkCudaErrors(cudaMalloc(&mesh->vertices, sizeof(float3) * vertices.size()));
    checkCudaErrors(cudaMemcpy(mesh->vertices, &vertices[0], vertices.size() * sizeof(float3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(&mesh->normals, sizeof(float3) * normals.size()));
    checkCudaErrors(cudaMemcpy(mesh->normals, &normals[0], normals.size() * sizeof(float3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(&mesh->triangles, sizeof(int) * triangles.size()));
    checkCudaErrors(cudaMemcpy(mesh->triangles, &triangles[0], triangles.size() * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(&mesh->triangleNormals, sizeof(int) * triangleNormals.size()));
    checkCudaErrors(cudaMemcpy(mesh->triangleNormals, &triangleNormals[0], triangleNormals.size() * sizeof(int), cudaMemcpyHostToDevice));
}

