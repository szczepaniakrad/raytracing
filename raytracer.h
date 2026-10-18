#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <memory>
#include <vector>
#include <iosfwd>


extern const double infinity;
extern const double pi;

double random_double();
double random_double(double min, double max);
double clamp(double x, double lo, double hi);

class Vec3 {
public:
    double e[3];

    Vec3();
    Vec3(double x, double y, double z);

    double x() const;
    double y() const;
    double z() const;

    Vec3   operator-() const;
    double operator[](int i) const;
    double& operator[](int i);

    Vec3& operator+=(const Vec3& v);
    Vec3& operator*=(double t);
    Vec3& operator/=(double t);

    double length() const;
    double length_squared() const;
    bool   near_zero() const;

    static Vec3 random();
    static Vec3 random(double min, double max);
};


Vec3 operator+(const Vec3& u, const Vec3& v);
Vec3 operator-(const Vec3& u, const Vec3& v);
Vec3 operator*(const Vec3& u, const Vec3& v);
Vec3 operator*(double t, const Vec3& v);
Vec3 operator*(const Vec3& v, double t);
Vec3 operator/(const Vec3& v, double t);

double dot(const Vec3& u, const Vec3& v);
Vec3   cross(const Vec3& u, const Vec3& v);
Vec3   unit_vector(const Vec3& v);

using Color = Vec3;

Vec3   random_in_unit_sphere();
Vec3   random_unit_vector();
Vec3   reflect(const Vec3& v, const Vec3& n);
Vec3   refract(const Vec3& uv, const Vec3& n, double etai_over_etat);
double schlick(double cosine, double ref_idx);


class Ray {
public:
    Vec3 orig;
    Vec3 dir;

    Ray();
    Ray(const Vec3& origin, const Vec3& direction);

    const Vec3& origin() const;
    const Vec3& direction() const;
    Vec3 at(double t) const;
};


class Material;

struct HitRecord {
    Vec3   p;
    Vec3   normal;
    double t;
    bool   front_face;
    std::shared_ptr<Material> mat;

    void set_face_normal(const Ray& r, const Vec3& outward_normal);
};

class Hittable {
public:
    virtual ~Hittable();
    virtual bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const = 0;
};

class Material {
public:
    virtual ~Material();
    virtual bool scatter(const Ray& r_in, const HitRecord& rec,
                         Color& attenuation, Ray& scattered) const = 0;


    virtual Color diffuse_albedo() const;
};

class Diffuse : public Material {
public:
    Color albedo;
    explicit Diffuse(const Color& a);
    bool scatter(const Ray& r_in, const HitRecord& rec,
                 Color& attenuation, Ray& scattered) const override;
    Color diffuse_albedo() const override;
};

class Mirror : public Material {
public:
    Color  albedo;
    double fuzz;
    Mirror(const Color& a, double f = 0.0);
    bool scatter(const Ray& r_in, const HitRecord& rec,
                 Color& attenuation, Ray& scattered) const override;
};

class Dielectric : public Material {
public:
    double ir;
    explicit Dielectric(double index_of_refraction);
    bool scatter(const Ray& r_in, const HitRecord& rec,
                 Color& attenuation, Ray& scattered) const override;
};

class Sphere : public Hittable {
public:
    Vec3   center;
    double radius;
    std::shared_ptr<Material> mat;
    Sphere(const Vec3& c, double r, std::shared_ptr<Material> m);
    bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const override;
};

class Plane : public Hittable {
public:
    Vec3   point;
    Vec3   normal;
    std::shared_ptr<Material> mat;
    Plane(const Vec3& p, const Vec3& n, std::shared_ptr<Material> m);
    bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const override;
};

class Triangle : public Hittable {
public:
    Vec3   v0, v1, v2;
    Vec3   normal;
    std::shared_ptr<Material> mat;
    Triangle(const Vec3& a, const Vec3& b, const Vec3& c, std::shared_ptr<Material> m);
    bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const override;
};


class Light {
public:
    Vec3  position;
    Color intensity;
    Light(const Vec3& p, const Color& i);
};

class Scene : public Hittable {
public:
    std::vector<std::shared_ptr<Hittable>> objects;
    std::vector<Light>                     lights;
    void add(std::shared_ptr<Hittable> object);
    void add_light(const Light& light);
    bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const override;
};

int  to_16bit(double c);
void write_color(std::ostream& out, const Color& pixel);

Color background(const Ray& r);
Color direct_light(const HitRecord& rec, const Scene& scene);
Color ray_color(const Ray& r, const Scene& scene, int depth);

#endif