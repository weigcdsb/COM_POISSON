


nu=4;
lam=200^nu;
pdf1 = com_pdf_robust(0:500, lam, nu);
pdf2 = com_pdf_sequentialZ(0:500, lam, nu);
figure(5)
plot(pdf1)
hold on
plot(pdf2)
hold off
[sum(~isfinite(pdf1)) sum(~isfinite(pdf2))]
sum((pdf1-pdf2).^2)