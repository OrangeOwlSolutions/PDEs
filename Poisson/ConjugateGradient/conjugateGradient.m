function x = conjugateGradient(A, b, tol)

% function [x] = conjgrad(A, b, x)
%     rsold = r' * r;
% 
%     for i = 1:length(b)
%     end
% end

x = b;
r = b - A * x;
%     p = r;
if norm(r) < tol
    return
end
y = -r;
z = A * y;
s = y' * z;
t = (r' * y) / s;
x = x + t * y;

for k = 1 : numel(b)
%         Ap = A * p;
%         alpha = rsold / (p' * Ap);
%         x = x + alpha * p;
%         r = r - alpha * Ap;
%         rsnew = r' * r;
%         if sqrt(rsnew) < 1e-10
%               break;
%         end
%         p = r + (rsnew / rsold) * p;
%         rsold = rsnew;
   r = r - t * z;
   if (norm(r) < tol)
        return;
   end
   B = (r' * z) / s;
   y = -r + B * y;
   z = A * y;
   s = y' * z;
   t = (r' * y) / s;
   x = x + t * y;
end
end
