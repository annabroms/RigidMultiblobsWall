
clear;
close all;

print_pngs = 1; 

%cfg = dlmread('../../Structures/Cylinder_N_86_Lg_1_9384_Rg_0_1484.vertex');
cfg = dlmread('../Structures/rt_optrod_aspect20_res1.vertex'); %distribution of blobs on the particle
cfg(1,:) = [];
cfg(cfg>1e4) = 0;

Np = 10;
dt = 0.001;
L = 2;
eta = 10;
t_dr = 1e-3/8*eta; %diffusive time-scale for rotational diffusion

simul_type = sprintf('../data/dynamic_rods_N%u',Np);
name = sprintf('%s/dt%1.5f_L%1.2f_eta%1.2f.random%u_L%1.2f.config',simul_type,dt,L,eta,Np,L);

%read in simulation data
%A = dlmread(['DP_run.config']);
%A = dlmread(['../data/dynamic_rods_T1_N10_movie/dt0.001_L1.00_tol001.random10_L1.00_tol001.config']);
%A = dlmread(['../data/dynamic_rods_T1_N10_movie/dt0.00100_L1.00_tol001.random10_L1.00_tol001.config']);
%A = dlmread(['../data/dynamic_rods_T1_N10_movie/dt0.00010_L1.00_tol001.random10_L1.00_tol001.config']);
A = dlmread(name); 
n_bods = round(A(1,1));
rem = mod(length(A),n_bods+1);
A(end-rem+1:end,:) = [];
A(1:n_bods+1:end,:) = [];


zmax = 0;
alist = [0.010838866643485, 0.007616276270953, 0.006207359652491, 0.004963143047909]; %given ar = 20
a = alist(1); %with res 1
%L = 16.18021593796416 %40.106052394096004
L = 2;

[sx,sy,sz] = sphere(20);
%[%X, Y] = meshgrid([0:0.5:L],[0:0.5:L]);
%[X, Y] = meshgrid([-L:0.5:L],[-L:0.5:L]);
f= 0; %frame number

skip = 10;
skip = 1; 
plot_walls=0;
end_ind = length(A)/n_bods;
end_ind = 200; 

%for i = (length(A)/n_bods) %1:skip:
for i = 1:skip:end_ind %1:skip:
    clf
    i

    f = f+1;
    
    % read coordinates 
    x = A((i-1)*n_bods+1:i*n_bods,1);
    y = A((i-1)*n_bods+1:i*n_bods,2);
    z = A((i-1)*n_bods+1:i*n_bods,3);
    s = A((i-1)*n_bods+1:i*n_bods,4);
    p = A((i-1)*n_bods+1:i*n_bods,5:end);


    r_vecs = [];
    for j = 1:length(x)
        R = Rot_From_Q(s(j),p(j,:));
        Bod = R*cfg' + [x(j);y(j);z(j)]*ones(1,length(cfg));
        r_vecs = [r_vecs Bod];
        for k = 1:length(Bod)
            xyz = Bod(:,k);
            h = surface(xyz(1)+a*sx,xyz(2)+a*sy,xyz(3)+a*sz,'facecolor','r','edgecolor','none');
            set(h,'FaceLighting','gouraud',...%'facealpha',0.2, ...
            'AmbientStrength',0.5, ...
            'DiffuseStrength',0.3, ...
            'Clipping','off',...
            'BackFaceLighting','lit', ...
            'SpecularStrength',0.3, ...
            'SpecularColorReflectance',0.7, ...
            'SpecularExponent',1)
            hold all
        end
     end



        %daspect([1 1 1])
        axis equal
        %view([20 7]) %view([-20 35]) %view([-140 10])%
        xlim([0 L])
        ylim([0 L])
        zlim([0 L])
%         xlim([-0.5*L L])
%         ylim([-0.5*L L])
%         zlim([-0.5*L L])
        %view([32,-19]); 
        view([40,22]);
        grid on
    
        %zlim([-a 4*a+a])
        hold all
        
        xaxisproperties= get(gca, 'XAxis');
        xaxisproperties.TickLabelInterpreter = 'latex'; % latex for x-axis
        yaxisproperties= get(gca, 'YAxis');
        yaxisproperties.TickLabelInterpreter = 'latex'; % latex for x-axis
        zaxisproperties= get(gca, 'ZAxis');
        zaxisproperties.TickLabelInterpreter = 'latex'; % latex for x-axis
        xlabel('$x$','interpreter','latex')
        ylabel('$y$','interpreter','latex')
        zlabel('$z$','interpreter','latex')
        
        time_str = sprintf('t = %1.5f ms',1000*f*skip*dt*t_dr);
        title(time_str,'interpreter','latex');
        
    if plot_walls
        surface(X,Y,0*X,'facecolor','k','edgecolor','none')
        hold all
        surface(X,Y,0*X+zmax,'facecolor','k','edgecolor','none','facealpha',0.1)
        hold all
    end

    camlight
    drawnow

    hold off
    
    %use export fig here instead or change resolution from the -r300
    %command to maybe r200
    if print_pngs == 1
        save_str = sprintf('../rods_pngs/rodstest%u_',Np);
        print('-dpng',[save_str num2str(f) '.png'],'-r100')
    end
end

function R = Rot_From_Q(s,p)
    P = [0, -1*p(3), p(2)
        p(3), 0, -1*p(1)
        -1*p(2), p(1), 0];
    R = 2*((p'*p) + (s^2-0.5)*eye(3) + s*P);
end
