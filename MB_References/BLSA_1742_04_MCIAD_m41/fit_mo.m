function F=fit_mo(P,Sexp,TE)

Sth(1,:)=P(3)+(P(1)*exp(-TE./P(2)));
F=Sexp-Sth';

end