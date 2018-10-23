clear all
close all
clc

M               = 256;
N               = 256;
xM              = 2;
yM              = 2;
Deltax          = xM / (M + 1);
Deltay          = yM / (N + 1);
x               = 0 : Deltax : xM;
y               = 0 : Deltay : yM;
[X, Y]          = meshgrid(x, y);

F               = ones(M * N, 1);

% A               = spdiags(B, d, m, n);

n               = 10;
e               = ones(n * n, 1);
A               = spdiags([-e -e 4*e -e -e], [-n -1 0 1 n], n * n, n * n)

K               = laplaceEqn(2, n);

J1=10;
h1=2/J1;
a1=sparse(1:(J1-1)^2,1:(J1-1)^2,4*ones(1,(J1-1)^2),(J1-1)^2,(J1-1)^2);
a2=sparse(J1:J1^2-2*J1+1,1:J1^2-3*J1+2,-ones(1,J1^2-3*J1+2),(J1-1)^2,(J1-1)^2);
for i=1:J1^2-2*J1
 s(i)=-1;
end
for i=1:J1-2
 s(i*(J1-1))=0;
end %sub-diagonal zero elements are considered when establishing s
a3=sparse(2:J1^2-2*J1+1,1:J1^2-2*J1,s,(J1-1)^2,(J1-1)^2);
A1=a1+a2+a2'+a3+a3';
