function F=fit_bi(P,Sexp,TE)

Sth(1,:)=P(5)+(P(1)*(P(2)*exp(-TE./P(3))+(1-P(2))*exp(-TE./P(4))));
F=Sexp-Sth';

end