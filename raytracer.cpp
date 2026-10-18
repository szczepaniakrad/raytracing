#include "raytracer.h"
#include <cmath>
#include <limits>
#include <random>
#include <fstream>
#include <sstream>


const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.14159265358979323846;

double random_double() {
    static thread_local std::mt19937 generator(std::random_device{}());
    static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(generator);
}
double random_double(double min, double max) {
    return min + (max - min) * random_double();
}
double clamp(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}


Vec3::Vec3() : e{0, 0, 0} {}
Vec3::Vec3(double x, double y, double z) : e{x, y, z} {}

double Vec3::x() const { return e[0]; }
double Vec3::y() const { return e[1]; }
double Vec3::z() const { return e[2]; }

Vec3   Vec3::operator-() const            { return Vec3(-e[0], -e[1], -e[2]); }
double Vec3::operator[](int i) const      { return e[i]; }
double& Vec3::operator[](int i)           { return e[i]; }

Vec3& Vec3::operator+=(const Vec3& v) { e[0]+=v.e[0]; e[1]+=v.e[1]; e[2]+=v.e[2]; return *this; }
Vec3& Vec3::operator*=(double t)      { e[0]*=t; e[1]*=t; e[2]*=t; return *this; }
Vec3& Vec3::operator/=(double t)      { return *this *= 1.0 / t; }

double Vec3::length()         const { return std::sqrt(length_squared()); }
double Vec3::length_squared() const { return e[0]*e[0] + e[1]*e[1] + e[2]*e[2]; }

bool Vec3::near_zero() const {
    const double s = 1e-8;
    return (std::fabs(e[0]) < s) && (std::fabs(e[1]) < s) && (std::fabs(e[2]) < s);
}

Vec3 Vec3::random() {
    return Vec3(random_double(), random_double(), random_double());
}
Vec3 Vec3::random(double min, double max) {
    return Vec3(random_double(min,max), random_double(min,max), random_double(min,max));
}


Vec3 operator+(const Vec3& u, const Vec3& v) { return Vec3(u.e[0]+v.e[0], u.e[1]+v.e[1], u.e[2]+v.e[2]); }
Vec3 operator-(const Vec3& u, const Vec3& v) { return Vec3(u.e[0]-v.e[0], u.e[1]-v.e[1], u.e[2]-v.e[2]); }

Vec3 operator*(const Vec3& u, const Vec3& v) { return Vec3(u.e[0]*v.e[0], u.e[1]*v.e[1], u.e[2]*v.e[2]); }
Vec3 operator*(double t, const Vec3& v)      { return Vec3(t*v.e[0], t*v.e[1], t*v.e[2]); }
Vec3 operator*(const Vec3& v, double t)      { return t * v; }
Vec3 operator/(const Vec3& v, double t)      { return (1.0/t) * v; }

double dot(const Vec3& u, const Vec3& v) { return u.e[0]*v.e[0] + u.e[1]*v.e[1] + u.e[2]*v.e[2]; }
Vec3 cross(const Vec3& u, const Vec3& v) {
    return Vec3(u.e[1]*v.e[2] - u.e[2]*v.e[1],
                u.e[2]*v.e[0] - u.e[0]*v.e[2],
                u.e[0]*v.e[1] - u.e[1]*v.e[0]);
}
Vec3 unit_vector(const Vec3& v) { return v / v.length(); }


Vec3 random_in_unit_sphere() {
    while (true) {
        Vec3 p = Vec3::random(-1.0, 1.0);
        if (p.length_squared() <= 1.0) return p;
    }
}
Vec3 random_unit_vector() { return unit_vector(random_in_unit_sphere()); }


Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - 2.0 * dot(v, n) * n;
}

Vec3 refract(const Vec3& uv, const Vec3& n, double etai_over_etat) {
    double cos_theta = std::fmin(dot(-uv, n), 1.0);
    Vec3 r_out_perp     = etai_over_etat * (uv + cos_theta * n);
    Vec3 r_out_parallel = -std::sqrt(std::fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

double schlick(double cosine, double ref_idx) {
    double r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * std::pow(1.0 - cosine, 5.0);
}


Ray::Ray() {}
Ray::Ray(const Vec3& origin, const Vec3& direction) : orig(origin), dir(direction) {}

const Vec3& Ray::origin()    const { return orig; }
const Vec3& Ray::direction() const { return dir; }
Vec3        Ray::at(double t) const { return orig + t * dir; }


void HitRecord::set_face_normal(const Ray& r, const Vec3& outward_normal) {
    front_face = dot(r.direction(), outward_normal) < 0.0;
    normal = front_face ? outward_normal : -outward_normal;
}

Hittable::~Hittable() = default;


Material::~Material() = default;


Color Material::diffuse_albedo() const { return Color(0, 0, 0); }


Diffuse::Diffuse(const Color& a) : albedo(a) {}

Color Diffuse::diffuse_albedo() const { return albedo; }

bool Diffuse::scatter(const Ray&, const HitRecord& rec,
                      Color& attenuation, Ray& scattered) const {
    Vec3 scatter_direction = rec.normal + random_unit_vector();
    if (scatter_direction.near_zero()) scatter_direction = rec.normal;
    scattered   = Ray(rec.p, scatter_direction);
    attenuation = albedo;
    return true;
}

Mirror::Mirror(const Color& a, double f) : albedo(a), fuzz(f < 1.0 ? f : 1.0) {}

bool Mirror::scatter(const Ray& r_in, const HitRecord& rec,
                     Color& attenuation, Ray& scattered) const {
    Vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    scattered   = Ray(rec.p, reflected + fuzz * random_in_unit_sphere());
    attenuation = albedo;
    return dot(scattered.direction(), rec.normal) > 0.0;
}

Dielectric::Dielectric(double index_of_refraction) : ir(index_of_refraction) {}

bool Dielectric::scatter(const Ray& r_in, const HitRecord& rec,
                         Color& attenuation, Ray& scattered) const {
    attenuation = Color(1.0, 1.0, 1.0);
    double refraction_ratio = rec.front_face ? (1.0 / ir) : ir;

    Vec3   unit_direction = unit_vector(r_in.direction());
    double cos_theta = std::fmin(dot(-unit_direction, rec.normal), 1.0);
    double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);

    bool cannot_refract = refraction_ratio * sin_theta > 1.0;
    Vec3 direction;
    if (cannot_refract || schlick(cos_theta, refraction_ratio) > random_double())
        direction = reflect(unit_direction, rec.normal);
    else
        direction = refract(unit_direction, rec.normal, refraction_ratio);

    scattered = Ray(rec.p, direction);
    return true;
}


Sphere::Sphere(const Vec3& c, double r, std::shared_ptr<Material> m)
    : center(c), radius(r), mat(std::move(m)) {}

bool Sphere::hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const {
    Vec3   oc     = r.origin() - center;
    double a      = r.direction().length_squared();
    double half_b = dot(oc, r.direction());
    double c      = oc.length_squared() - radius * radius;
    double disc   = half_b * half_b - a * c;
    if (disc < 0.0) return false;
    double sqrtd  = std::sqrt(disc);

    double root = (-half_b - sqrtd) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || root > t_max) return false;
    }

    rec.t = root;
    rec.p = r.at(root);
    Vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat = mat;
    return true;
}

Plane::Plane(const Vec3& p, const Vec3& n, std::shared_ptr<Material> m)
    : point(p), normal(unit_vector(n)), mat(std::move(m)) {}

bool Plane::hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const {
    double denom = dot(normal, r.direction());
    if (std::fabs(denom) < 1e-8) return false;
    double t = dot(point - r.origin(), normal) / denom;
    if (t < t_min || t > t_max) return false;

    rec.t = t;
    rec.p = r.at(t);
    rec.set_face_normal(r, normal);
    rec.mat = mat;
    return true;
}

Triangle::Triangle(const Vec3& a, const Vec3& b, const Vec3& c, std::shared_ptr<Material> m)
    : v0(a), v1(b), v2(c), mat(std::move(m)) {
    normal = unit_vector(cross(v1 - v0, v2 - v0));
}


bool Triangle::hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const {
    const double eps = 1e-8;
    Vec3   edge1 = v1 - v0;
    Vec3   edge2 = v2 - v0;
    Vec3   pvec  = cross(r.direction(), edge2);
    double det   = dot(edge1, pvec);
    if (std::fabs(det) < eps) return false;
    double inv_det = 1.0 / det;


    Vec3   tvec = r.origin() - v0;
    double u    = dot(tvec, pvec) * inv_det;
    if (u < 0.0 || u > 1.0) return false;

    Vec3   qvec = cross(tvec, edge1);
    double v    = dot(r.direction(), qvec) * inv_det;
    if (v < 0.0 || u + v > 1.0) return false;

    double t = dot(edge2, qvec) * inv_det;
    if (t < t_min || t > t_max) return false;

    rec.t = t;
    rec.p = r.at(t);
    rec.set_face_normal(r, normal);
    rec.mat = mat;
    return true;
}

Light::Light(const Vec3& p, const Color& i) : position(p), intensity(i) {}

void Scene::add(std::shared_ptr<Hittable> object) { objects.push_back(std::move(object)); }

void Scene::add_light(const Light& light) { lights.push_back(light); }

bool Scene::hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const {
    HitRecord temp;
    bool   hit_anything   = false;
    double closest_so_far = t_max;
    for (const auto& object : objects) {
        if (object->hit(r, t_min, closest_so_far, temp)) {
            hit_anything   = true;
            closest_so_far = temp.t;
            rec            = temp;
        }
    }
    return hit_anything;
}

int to_16bit(double c) {
    c = std::sqrt(clamp(c, 0.0, 1.0));
    return static_cast<int>(std::lround(65535.0 * c));
}
void write_color(std::ostream& out, const Color& pixel) {
    out << to_16bit(pixel.x()) << ' '
        << to_16bit(pixel.y()) << ' '
        << to_16bit(pixel.z()) << '\n';
}

Color background(const Ray& r) {
    Vec3   unit_direction = unit_vector(r.direction());
    double a = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - a) * Color(1.0, 1.0, 1.0) + a * Color(0.5, 0.7, 1.0);
}


Color direct_light(const HitRecord& rec, const Scene& scene) {
    Color result(0, 0, 0);


    Color albedo = rec.mat->diffuse_albedo();
    if (albedo.near_zero()) return result;

    for (const auto& light : scene.lights) {
        Vec3   to_light = light.position - rec.p;
        double dist     = to_light.length();
        Vec3   l_dir    = to_light / dist;

        double cos_theta = dot(rec.normal, l_dir);
        if (cos_theta <= 0.0) continue;


        Ray       shadow_ray(rec.p, l_dir);
        HitRecord shadow_rec;
        if (scene.hit(shadow_ray, 0.001, dist - 0.001, shadow_rec)) continue;


        result += albedo * light.intensity * (cos_theta / (dist * dist));
    }
    return result;
}

Color ray_color(const Ray& r, const Scene& scene, int depth) {
    if (depth <= 0) return Color(0, 0, 0);

    HitRecord rec;
    if (scene.hit(r, 0.001, infinity, rec)) {
        Color direct = direct_light(rec, scene);

        Ray   scattered;
        Color attenuation;
        if (rec.mat->scatter(r, rec, attenuation, scattered))
            return direct + attenuation * ray_color(scattered, scene, depth - 1);
        return direct;
    }
    return background(r);
}

static Scene build_scene() {
    Scene scene;

    auto ground = std::make_shared<Diffuse>(Color(0.55, 0.55, 0.6));
    scene.add(std::make_shared<Plane>(Vec3(0, -0.5, 0), Vec3(0, 1, 0), ground));

    scene.add(std::make_shared<Sphere>(Vec3(-2.2, 0.0, 4.5), 0.5,
              std::make_shared<Mirror>(Color(0.85, 0.85, 0.90))));
    scene.add(std::make_shared<Sphere>(Vec3(-0.7, 0.0, 4.5), 0.5,
              std::make_shared<Mirror>(Color(0.90, 0.70, 0.30))));
    scene.add(std::make_shared<Sphere>(Vec3( 0.7, 0.0, 4.5), 0.5,
              std::make_shared<Mirror>(Color(0.80, 0.50, 0.40))));
    scene.add(std::make_shared<Sphere>(Vec3( 2.2, 0.0, 4.5), 0.5,
              std::make_shared<Mirror>(Color(0.50, 0.60, 0.85), 0.08)));

    scene.add(std::make_shared<Sphere>(Vec3(-1.2, -0.1, 2.7), 0.4,
              std::make_shared<Diffuse>(Color(0.80, 0.20, 0.20))));
    scene.add(std::make_shared<Sphere>(Vec3( 1.2, -0.1, 2.7), 0.4,
              std::make_shared<Diffuse>(Color(0.20, 0.70, 0.35))));
    scene.add(std::make_shared<Sphere>(Vec3( 0.0, -0.1, 2.7), 0.4,
              std::make_shared<Dielectric>(1.5)));


    scene.add(std::make_shared<Triangle>(
              Vec3(-3.0, -0.5, 3.4), Vec3(-1.6, -0.5, 3.0), Vec3(-2.3, 1.1, 3.2),
              std::make_shared<Diffuse>(Color(0.85, 0.45, 0.15))));


    scene.add_light(Light(Vec3(-3.5, 6.0, 1.5), Color(70.0, 64.0, 54.0)));
    scene.add_light(Light(Vec3( 3.5, 5.0, 0.5), Color(24.0, 30.0, 46.0)));

    return scene;
}

int main() {
    const int nx = 800;
    const int ny = 400;
    const int samples_per_pixel = 50;
    const int max_depth = 50;

    const double hx = 4.0;
    const double hy = 2.0;
    const double d  = 1.0;
    const Vec3   E(0, 1, -2);
    const double px = hx / (nx - 1);
    const double py = hy / (ny - 1);

    Scene scene = build_scene();

    std::vector<Color> framebuffer(static_cast<size_t>(nx) * ny);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            Color pixel(0, 0, 0);
            for (int s = 0; s < samples_per_pixel; ++s) {
                double rx = (samples_per_pixel > 1) ? (random_double() - 0.5) : 0.0;
                double ry = (samples_per_pixel > 1) ? (random_double() - 0.5) : 0.0;
                double x = -hx / 2.0 + (i + rx) * px;
                double y =  hy / 2.0 - (j + ry) * py;
                Vec3 P(E.x() + x, E.y() + y, E.z() + d);
                Ray  r(E, unit_vector(P - E));
                pixel += ray_color(r, scene, max_depth);
            }
            framebuffer[static_cast<size_t>(j) * nx + i] = pixel / samples_per_pixel;
        }
    }

    std::ofstream out("scena.ppm");
    out << "P3\n" << nx << ' ' << ny << "\n65535\n";
    std::ostringstream buf;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            write_color(buf, framebuffer[static_cast<size_t>(j) * nx + i]);
    out << buf.str();
    out.close();

    return 0;
}