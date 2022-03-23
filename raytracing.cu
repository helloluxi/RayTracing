
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <helper_cuda.h>
#include <helper_math.h>
#include <curand_kernel.h>

#define EPSILON 1.0e-7f
#define PI 3.141592654f

__device__ float3 tangentToWorld(float3 vector, float3 normal) {
    float3 right = abs(normal.z) > 1 - EPSILON ? 
        make_float3(1, 0, 0) : 
        normalize(make_float3(normal.y, -normal.x, 0));
    float3 forward = cross(normal, right);
    return vector.x * right + vector.y * forward + vector.z * normal;
}

__device__ float3 sampleHemisphere(float3 normal, curandState* rd) {
    float cos2Theta = curand_uniform(rd), 
        cosTheta = sqrt(cos2Theta),
        sinTheta = sqrt(1 - cos2Theta), 
        phi = curand_uniform(rd) * PI * 2;
    return tangentToWorld(make_float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta), normal);
}

__device__ float3 sampleGGX(float3 normal, float alpha2, curandState* rd) {
    float E = curand_uniform(rd);
    float cos2Theta = alpha2 == 0 ? 1 : (1 - E) / (1 - E * (1 - alpha2));
    float cosTheta = sqrt(cos2Theta);
    float sinTheta = sqrt(1 - cos2Theta);
    float phi = curand_uniform(rd) * PI * 2;
    /*float d = 1 - cos2Theta * (1 - alpha2);
    float pdf = alpha2 * cosTheta / (PI * d * d);*/
    return tangentToWorld(make_float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta), normal);
}

__device__ inline float pow2(float f) { return f * f; }

__device__ inline float pow4(float f) { return pow2(pow2(f)); }

__device__ inline float pow5(float f) { return pow4(f) * f; }

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
        float alpha2 = pow4(roughness);
        float k = pow2(roughness) * 0.25f;
        return (metallic * alpha2 / (pow2(1 - pow2(NoH) * (1 - alpha2))
            * 4 * (NoL * (1 - k) + k) * (NoV * (1 - k) + k)) + (1 - metallic)) / PI * albedo;
    }
};
// struct Triangle {
//     float3 a, b, c, n;
//     int materialId;
//     Triangle(float3 a, float3 b, float3 c, int materialId) : a(a), b(b), c(c), n(cross(b - a, c - a)), materialId(materialId) {}
//     __device__ Intersect getIntersection(Ray* ray) {
//         Intersect intersect{};
//         float3 ac = c - a, ab = b - a, ao = ray->origin - a, pvec = cross(ray->direction, ac), qvec = cross(ao, ab);
//         float inv_det = 1.f / dot(ab, pvec), u = dot(ao, pvec) * inv_det, v = dot(ray->direction, qvec) * inv_det;
//         if (dot(ray->direction, n) > 0 || u < 0 || v < 0 || u + v > 1)
//             return intersect;
//         intersect.exist = true;
//         intersect.point = a + ab * v + ac * u;
//         intersect.normal = n;
//         intersect.distance = dot(ao, n);
//         intersect.materialId = materialId;
//         return intersect;
//     }
// };
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
    int materialCount, planeCount, sphereCount/*, triangleCount*/;
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
            //             pow2(scene->light->size / lightIntersect.distance) * scale *
            //             scene->light->intensity * mat.BRDF(intersect.normal, -ray.direction, -rayFromLight.direction);
            //     }
            // }

            float3 V = -ray.direction, L;
            if (curand_uniform(rd) < 1 - mat.metallic) {
                L = sampleHemisphere(intersect.normal, rd);
                // float3 H = normalize(V + L);
                // float HoL = dot(H, L);
                // float OneMinusFD90 = 0.5f - 2 * mat.roughness * HoL * HoL;
                // float F = (1 - OneMinusFD90 * pow5(1 - dot(intersect.normal, V))) * (1 - OneMinusFD90 * pow5(1 - dot(intersect.normal, L)));
                scale *= mat.albedo;
            }
            else {
                float3 H = sampleGGX(intersect.normal, pow4(mat.roughness), rd);
                L = reflect(ray.direction, H);
                float NoV = saturate(dot(intersect.normal, V));
                float NoH = saturate(dot(intersect.normal, H));
                float VoH = saturate(dot(V, H));
                float NoL = saturate(dot(intersect.normal, L));
                //float OneMinusVoH5 = pow5(1 - VoH);
                //float3 F = mat.metallicAlbedo * (1 - OneMinusVoH5) + OneMinusVoH5;
                float k = pow2(mat.roughness) * 0.25f;
                float denominator = NoH * (NoL * (1 - k) + k) * (NoV * (1 - k) + k);
                if(denominator == 0) break;
                scale *= VoH * NoL / denominator * mat.albedo;
            }
            ray.direction = L;
            ray.origin = intersect.point;
        }
    }
    return color;
}

__global__ void init(curandState* rd) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(idx, idx, idx, &rd[idx]);
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

void runInit(curandState* rd, int gridNum, int threadNum){
    init <<<gridNum, threadNum>>> (rd);
}

void runKernel(float3* result, curandState* rd, Scene scene,
    float3 cameraPos, float3 cameraFront, float3 cameraUp, float3 cameraRight,
    int screenWidth, int screenHeight, int nSample, int maxDepth, float tangentPerPixel,
    int gridNum, int threadNum){
    kernel<<<gridNum, threadNum>>>(result, rd, scene,
        cameraPos, cameraFront, cameraUp, cameraRight,
        screenWidth, screenHeight, nSample, maxDepth, tangentPerPixel);
}