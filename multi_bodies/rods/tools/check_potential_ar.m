% Define the center and radii of the spheroid
close all
clear 

xc = 0;
yc = 0;
zc = 0;
% Parameters for the potential
a = 0.3;
b = a/5;

% a = 3;
% b = 2;
rx = b;
ry = b;
rz = a;

% Generate the spheroid
n = 20; % number of points to use in each dimension
[x,y,z] = ellipsoid(xc,yc,zc,rx,ry,rz,n);

% Plot the spheroid
figure
surf(x,y,z,'FaceColor','r','FaceAlpha',0.2)
axis equal
%%
a = 0.25;
b = a/10;
r = b;
h = 2*a-2*b;

% Generate the cylinder
n = 20; % number of points to use in each dimension
[x,y,z] = cylinder(r,n);

% Scale the cylinder to the desired height
z = z * h-a+b;

% Plot the cylinder
hold on
surf(x,y,z,'FaceAlpha',0.2)
axis equal