
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <helper_cuda.h>
#include <helper_math.h>
#include <curand_kernel.h>

#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

#define EPSILON 1.0e-7f
#define PI 3.141592654f

const float3 cameraPos = make_float3(0, -15, 5);
const float3 cameraFront = make_float3(0, 1, 0);
const float3 cameraUp = make_float3(0, 0, 1);
const float3 cameraRight = make_float3(1, 0, 0);

const float verticalFOV = 1.0f;
//const float haltChance = .9f;
const float gammaCorrector = 2.2f;
const int screenWidth = 1024;
const int screenHeight = 576;
const int nSample = 4096;
const int maxDepth = 6;
const int myThreadNum = 512;

const float invGamma = 1.0f / gammaCorrector;
const int pixelNum = screenWidth * screenHeight;
const int myGridNum = (pixelNum + myThreadNum - 1) / myThreadNum;

__device__ float3 tangentToWorld(float3 vector, float3 normal) {
    float3 right = abs(normal.z) > 1 - EPSILON ? make_float3(1, 0, 0) : normalize(make_float3(normal.y, -normal.x, 0));
    float3 forward = cross(normal, right);
    return vector.x * right + vector.y * forward + vector.z * normal;
}

__device__ float3 sampleHemisphere(float3 normal, curandState* rd) {
    float cos2Theta = curand_uniform(rd), cosTheta = sqrt(cos2Theta),
        sinTheta = sqrt(1 - cos2Theta), phi = curand_uniform(rd) * PI * 2;
    return tangentToWorld(make_float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta), normal);
}

__device__ float3 sampleGGX(float3 normal, float alpha2, curandState* rd) {
    float E = curand_uniform(rd);
    float cos2Theta = (1 - E) / (1 - E * (1 - alpha2));
    float cosTheta = sqrt(cos2Theta);
    float sinTheta = sqrt(1 - cos2Theta);
    float phi = curand_uniform(rd) * PI * 2;
    /*float d = 1 - cos2Theta * (1 - alpha2);
    float pdf = alpha2 * cosTheta / (PI * d * d);*/
    return tangentToWorld(make_float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta), normal);
}

__device__ inline float Pow2(float f) { return f * f; }

__device__ inline float Pow4(float f) { return Pow2(Pow2(f)); }

__device__ inline float Pow5(float f) { return Pow4(f) * f; }

struct Ray { float3 origin, direction; };
struct Intersect { bool exist; float distance; float3 point, normal; int materialId; };
struct Shader { 
    float3 albedo; 
    float metallic, roughness;
    __device__ float3 BRDF(float3 N, float3 V, float3 L){
        float3 H = normalize(V + L);
        float NoV = saturate(dot(N, V));
        float NoH = saturate(dot(N, H));
        float VoH = saturate(dot(V, H));
        float NoL = saturate(dot(N, L));
        float alpha2 = Pow4(roughness);
        float k = Pow2(roughness) * 0.25f;
        return (metallic * alpha2 / (Pow2(1 - Pow2(NoH) * (1 - alpha2))
            * 4 * (NoL * (1 - k) + k) * (NoV * (1 - k) + k)) + (1 - metallic)) / PI * albedo;
    }
};
/*struct Triangle {
    float3 a, b, c, n;
    int materialId;
    Triangle(float3 a, float3 b, float3 c, int materialId) : a(a), b(b), c(c), n(cross(b - a, c - a)), materialId(materialId) {}
    __device__ Intersect getIntersection(Ray* ray) {
        Intersect intersect{};
        float3 ac = c - a, ab = b - a, ao = ray->origin - a, pvec = cross(ray->direction, ac), qvec = cross(ao, ab);
        float inv_det = 1.f / dot(ab, pvec), u = dot(ao, pvec) * inv_det, v = dot(ray->direction, qvec) * inv_det;
        if (dot(ray->direction, n) > 0 || u < 0 || v < 0 || u + v > 1)
            return intersect;
        intersect.exist = true;
        intersect.point = a + ab * v + ac * u;
        intersect.normal = n;
        intersect.distance = dot(ao, n);
        intersect.materialId = materialId;
        return intersect;
    }
};*/
struct Plane {
    float3 point, normal;
    int materialId;
    __device__ Intersect getIntersection(Ray* ray) {
        Intersect intersect{};
        float cosIn = -dot(ray->direction, this->normal);
        float3 pa = this->point - ray->origin;
        float t = -dot(pa, this->normal) / cosIn;
        if (cosIn < EPSILON || t < EPSILON)
            return intersect;
        intersect.exist = true;
        intersect.distance = t;
        intersect.point = ray->origin + t * ray->direction;
        intersect.normal = this->normal;
        intersect.materialId = this->materialId;
        return intersect;
    }
};
struct Sphere {
    float3 center;
    float r;
    int materialId;
    __device__ Intersect getIntersection(Ray* ray) {
        Intersect intersect{};
        float3 pc = ray->origin - this->center;
        float proj = -dot(ray->direction, pc);
        float lengthInSphereSquared = this->r * this->r - (dot(pc, pc) - proj * proj);
        float lengthInSphere = sqrt(lengthInSphereSquared);
        if (lengthInSphereSquared < 0 || proj < lengthInSphere)
            return intersect;
        float t = proj - lengthInSphere;
        intersect.exist = true;
        intersect.distance = t;
        intersect.point = ray->origin + t * ray->direction;
        intersect.normal = (intersect.point - this->center) / this->r;
        intersect.materialId = this->materialId;
        return intersect;
    }
};
struct Texture {
    float3* texture;
    int size[3];
    inline __device__ float3 at(int x, int y) { return texture[x + y * size[0]]; }
    __device__ float3 interpolate(float u, float v) {
        u = clamp((u + 1) * 0.5f, -1.0f, 1.0f) * (size[0] - 1);
        v = clamp((v + 1) * 0.5f, -1.0f, 1.0f) * (size[1] - 1);
        int x1 = (int)floor(u), x2 = (int)ceil(u), y1 = (int)floor(v), y2 = (int)ceil(v);
        return (at(x1, y1) * (y2 - v) + at(x1, y2) * (v - y1)) * (x2 - u)
            + (at(x2, y1) * (y2 - v) + at(x2, y2) * (v - y1)) * (u - x1);
    }
};
struct AreaLight{
    float3 center, intensity;
    float3 right, up, normal;
    float size;
    __device__ float3 sample(curandState* rd){
        return center + ((curand_uniform(rd) - 0.5f) * right + (curand_uniform(rd) - 0.5f) * up) * size;
    }
    __device__ Intersect getIntersection(Ray* ray){
        Intersect intersect{};
        float cosIn = -dot(ray->direction, this->normal);
        float3 pa = this->center - ray->origin;
        float t = -dot(pa, this->normal) / cosIn;
        float3 hitPos = ray->origin + t * ray->direction, hitRelPos = hitPos - this->center;
        float x = dot(hitRelPos, right), y = dot(hitRelPos, up);
        if (cosIn < EPSILON || t < EPSILON || max(abs(x), abs(y)) * 2 > size)
            return intersect;
        intersect.exist = true;
        intersect.distance = t;
        intersect.point = hitPos;
        intersect.normal = this->normal;
        intersect.materialId = -1;
        return intersect;
    }
};
struct Scene
{
    Shader* materials;
    Plane* planes;
    Sphere* spheres;
    AreaLight* light;
    // Triangle* triangles;
    bool useLightSampling;
    int planeCount, sphereCount/*, triangleCount*/;
    Texture* envPX, * envPY, * envPZ, * envMX, * envMY, * envMZ;
    __device__ Intersect findIntersection(Ray* ray) {
        Intersect current{
            false,
            __FLT_MAX__
        };
        if(useLightSampling){
            auto intersect = light->getIntersection(ray);
            if (intersect.exist && intersect.distance < current.distance)
                current = intersect;
        }
        for (int i = 0; i < this->planeCount; ++i)
        {
            auto intersect = this->planes[i].getIntersection(ray);
            if (intersect.exist && intersect.distance < current.distance)
                current = intersect;
        }
        for (int i = 0; i < this->sphereCount; ++i)
        {
            auto intersect = this->spheres[i].getIntersection(ray);
            if (intersect.exist && intersect.distance < current.distance)
                current = intersect;
        }
        // for (int i = 0; i < this->triangleCount; ++i)
        // {
        //     auto intersect = this->triangles[i].getIntersection(ray);
        //     if (intersect.exist && intersect.distance < current.distance)
        //         current = intersect;
        // }
        return current;
    }
    __device__ float3 sampleEnvironment(float3 dir) {
        float absX = abs(dir.x), absY = abs(dir.y), absZ = abs(dir.z);
        if (absX >= absY && absX >= absZ)
            return dir.x > 0 ?
            this->envPX->interpolate(-dir.y / dir.x, -dir.z / dir.x) :
            this->envMX->interpolate(-dir.y / dir.x, dir.z / dir.x);
        else if (absY >= absX && absY >= absZ)
            return dir.y > 0 ?
            this->envPY->interpolate(dir.x / dir.y, -dir.z / dir.y) :
            this->envMY->interpolate(dir.x / dir.y, dir.z / dir.y);
        else
            return dir.z > 0 ?
            this->envPZ->interpolate(dir.x / dir.z, dir.y / dir.z) :
            this->envMZ->interpolate(dir.x / dir.z, dir.y / dir.z);
    }
};

__device__ float3 trace(Ray* primaryRay, Scene* scene, curandState* rd, int maxDepth){
    Ray ray = *primaryRay;
    Intersect intersect{};
    float3 scale = make_float3(1), color{};
    for (int depth = 0; depth < maxDepth; ++depth) {
        intersect = scene->findIntersection(&ray);
        if (!intersect.exist) {
            if(!scene->useLightSampling)
                color += scale * scene->sampleEnvironment(ray.direction);
            break;
        }
        else if(intersect.materialId == -1){
            // if(depth == 0)
                color += scale * scene->light->intensity;
            break;
        }
        else {
            Shader& mat = scene->materials[intersect.materialId];
            // if(scene->useLightSampling){
            //     float3 lightPos = scene->light->sample(rd);
            //     Ray rayFromLight{lightPos, normalize(intersect.point - lightPos)};
            //     Intersect lightIntersect = scene->findIntersection(&rayFromLight);
            //     if(length(lightIntersect.point - intersect.point) < EPSILON){
            //         color += saturate(dot(rayFromLight.direction, scene->light->normal)) *
            //             saturate(-dot(rayFromLight.direction, intersect.normal)) * 
            //             Pow2(scene->light->size / lightIntersect.distance) * scale *
            //             scene->light->intensity * mat.BRDF(intersect.normal, -ray.direction, -rayFromLight.direction);
            //     }
            // }

            float3 V = -ray.direction, L;
            if (curand_uniform(rd) < 1 - mat.metallic) {
                L = sampleHemisphere(intersect.normal, rd);
                // float3 H = normalize(V + L);
                // float HoL = dot(H, L);
                // float OneMinusFD90 = 0.5f - 2 * mat.roughness * HoL * HoL;
                // float F = (1 - OneMinusFD90 * Pow5(1 - dot(intersect.normal, V))) * (1 - OneMinusFD90 * Pow5(1 - dot(intersect.normal, L)));
                scale *= mat.albedo;
            }
            else {
                float3 H = sampleGGX(intersect.normal, Pow4(mat.roughness), rd);
                L = reflect(ray.direction, H);
                float NoV = saturate(dot(intersect.normal, V));
                float NoH = saturate(dot(intersect.normal, H));
                float VoH = saturate(dot(V, H));
                float NoL = saturate(dot(intersect.normal, L));
                //float OneMinusVoH5 = Pow5(1 - VoH);
                //float3 F = mat.metallicAlbedo * (1 - OneMinusVoH5) + OneMinusVoH5;
                float k = Pow2(mat.roughness) * 0.25f;
                scale *= VoH * NoL / (NoH * (NoL * (1 - k) + k) * (NoV * (1 - k) + k)) * mat.albedo;
                // return VoH * NoL / (NoH * (NoL * (1 - k) + k) * (NoV * (1 - k) + k)) * make_float3(1);
            }
            ray.direction = L;
            ray.origin = intersect.point;
        }
    }
    return color;
}

__global__ void init(curandState* rd) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(idx, idx, 0, &rd[idx]);
}

__global__ void kernel(float3* arr, curandState* rd, Scene scene,
    float3 cameraPos, float3 cameraFront, float3 cameraUp, float3 cameraRight,
    int screenWidth, int screenHeight, int nSample, int maxDepth, float tangentPerPixel)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int x = idx % screenWidth, y = idx / screenWidth;
    for(int sample = 0; sample < nSample; ++sample){
        Ray ray{ cameraPos, normalize(cameraFront
            + cameraRight * (x - screenWidth * 0.5f + curand_uniform(&rd[idx])) * tangentPerPixel
            + cameraUp * (screenHeight * 0.5f - y - curand_uniform(&rd[idx])) * tangentPerPixel) };
        arr[idx] += trace(&ray, &scene, &rd[idx], maxDepth);
    }
    arr[idx] /= nSample;
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

#define EXAMPLE_1
int main()
{
    auto awakeTime = clock();
    // init random seed
    curandState* rd;
    checkCudaErrors(cudaMalloc((void**)&rd, pixelNum * sizeof(curandState)));
    init <<<myGridNum, myThreadNum>>> (rd);
    checkCudaErrors(cudaDeviceSynchronize());

    // initscene
    Scene scene;
    #ifdef EXAMPLE_1
    checkCudaErrors(cudaMallocManaged(&scene.materials, sizeof(Shader) * 5));
    checkCudaErrors(cudaMallocManaged(&scene.planes, sizeof(Plane) * 6));
    checkCudaErrors(cudaMallocManaged(&scene.spheres, sizeof(Sphere) * 2));
    // checkCudaErrors(cudaMallocManaged(&scene.triangles, sizeof(Triangle) * 2));
    scene.materials[0] = Shader{ make_float3(1.0f, 1.0f, 1.0f), 0.0f, 0.5f };
    scene.materials[1] = Shader{ make_float3(1.0f, 0.0f, 0.0f), 0.0f, 0.5f };
    scene.materials[2] = Shader{ make_float3(0.0f, 1.0f, 0.0f), 0.0f, 0.5f };
    scene.materials[3] = Shader{ make_float3(1.0f), 0.2f, 0.5f };
    scene.materials[4] = Shader{ make_float3(1.0f), 1.0f, 0.2f };
    scene.planeCount = 6;
    scene.planes[0] = Plane{ make_float3(0, 0, 0), make_float3(0, 0, 1), 0 };
    scene.planes[1] = Plane{ make_float3(0, 0, 10), make_float3(0, 0, -1), 0 };
    scene.planes[2] = Plane{ make_float3(0, 10, 0), make_float3(0, -1, 0), 0 };
    scene.planes[3] = Plane{ make_float3(0, -10, 0), make_float3(0, 1, 0), 0 };
    scene.planes[4] = Plane{ make_float3(-10, 0, 0), make_float3(1, 0, 0), 1 };
    scene.planes[5] = Plane{ make_float3(10, 0, 0), make_float3(-1, 0, 0), 2 };
    scene.sphereCount = 2;
    scene.spheres[0] = Sphere{ make_float3(4, 0, 3), 3, 3 };
    scene.spheres[1] = Sphere{ make_float3(-4, 4, 3), 3, 4 };
    scene.useLightSampling = true;
    checkCudaErrors(cudaMallocManaged(&scene.light, sizeof(AreaLight)));
    scene.light[0] = AreaLight{ make_float3(0, 0, 9.99f), make_float3(5),
        make_float3(1, 0, 0), make_float3(0, 1, 0), make_float3(0, 0, -1), 5};
    // scene.triangleCount = 2;
    // float lightSize = 2;
    // scene.triangles[0] = Triangle(make_float3(-lightSize, lightSize, 9.99f),
    //     make_float3(-lightSize, -lightSize, 9.9f), make_float3(lightSize, lightSize, 9.99f), 5);
    // scene.triangles[1] = Triangle(make_float3(-lightSize, -lightSize, 9.99f),
    //     make_float3(lightSize, -lightSize, 9.99f), make_float3(lightSize, lightSize, 9.9f), 5);
    #else
    checkCudaErrors(cudaMallocManaged(&scene.materials, sizeof(Shader) * 3));
    checkCudaErrors(cudaMallocManaged(&scene.planes, sizeof(Plane) * 1));
    checkCudaErrors(cudaMallocManaged(&scene.spheres, sizeof(Sphere) * 2));
    scene.materials[0] = Shader{ make_float3(1), 0.5f, 0.1f };
    scene.materials[1] = Shader{ make_float3(1), 1.0f, 0.2f };
    scene.materials[2] = Shader{ make_float3(1), 1.0f, 0.5f };
    scene.planeCount = 1;
    scene.planes[0] = Plane{ make_float3(0), make_float3(0, 0, 1), 0 };
    scene.sphereCount = 2;
    scene.spheres[0] = Sphere{ make_float3(-3.5f, 0, 3), 3, 1 };
    scene.spheres[1] = Sphere{ make_float3(3.5f, 0, 3), 3, 2 };
    LoadTexture(scene.envMX, "texture/-X.png");
    LoadTexture(scene.envMY, "texture/-Y.png");
    LoadTexture(scene.envMZ, "texture/-Z.png");
    LoadTexture(scene.envPX, "texture/+X.png");
    LoadTexture(scene.envPY, "texture/+Y.png");
    LoadTexture(scene.envPZ, "texture/+Z.png");
    #endif

    // Malloc result image
    float3* result;
    checkCudaErrors(cudaMallocManaged(&result, sizeof(float3) * pixelNum));

    // Ray Tracing
    auto startTime = clock();
    kernel <<<myGridNum, myThreadNum>>> (result, rd, scene,
        cameraPos, cameraFront, cameraUp, cameraRight,
        screenWidth, screenHeight, nSample, maxDepth, verticalFOV / screenHeight);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("Initializing Time = %.3fs\n", (float)(startTime - awakeTime) / CLOCKS_PER_SEC);
    printf("Ray Tracing Time = %.3fs\n", (float)(clock() - startTime) / CLOCKS_PER_SEC);

    // Write image to file
    stbi_uc output[pixelNum * 3];
    for (size_t i = 0; i < pixelNum; i++)
    {
        output[i * 3] = (stbi_uc)(powf(clamp(result[i].x, 0.0f, 1.0f), invGamma) * 255.f);
        output[i * 3 + 1] = (stbi_uc)(powf(clamp(result[i].y, 0.0f, 1.0f), invGamma) * 255.f);
        output[i * 3 + 2] = (stbi_uc)(powf(clamp(result[i].z, 0.0f, 1.0f), invGamma) * 255.f);
    }
    stbi_write_png("render.png", screenWidth, screenHeight, 3, output, screenWidth * 3);

    checkCudaErrors(cudaDeviceReset());
    return 0;
}
