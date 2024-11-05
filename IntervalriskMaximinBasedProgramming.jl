using JuMP;
using Gurobi;
using CDDLib;
using HDF5;
#using Polyhedra;
using JLD2;
using JLD;
using CSV;
using DataFrames;
using MAT;
import LinearAlgebra: norm
##################################################################################################
#                       Input data and parameters
##################################################################################################
G = 12; Z = 4;
c = [13.32; 13.32; 20.7; 20.93; 26.11; 10.52; 10.52; 6.02; 5.47; 7; 10.52; 10.89];
cU = [1.68; 1.68; 3.30; 4.07; 1.89; 5.48; 5.48; 4.98; 5.53; 8.0; 3.48; 5.11]
cD = [2.32; 2.32; 4.67; 3.93; 3.11; 3.52; 3.52; 5.02; 4.97; 6.0; 2.52; 2.89];
pmax = 1.2*[106.4; 106.4; 245; 413.7; 42; 108.5; 108.5; 280; 280; 210; 217; 245];
pmin = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
Rmax = [48; 48; 84; 216; 42; 36; 36; 60; 60; 48; 72; 48];
D = [84; 75; 139; 58; 55; 106; 97; 132; 135; 150; 205; 150; 245; 77; 258; 141; 100];
W = [500   0      0       0;
0     500    0       0;
0     0      300     0;
0     0      0       300];

Uξ = [0.3999 0.4007 0.2719 0.3166];
Lξ = [-0.2313 -0.2305 -0.1705 -0.1110];
ξ = [0.2367 0.2325 0.1774 0.1321];
μ = [0.08437 0.085105 0.050705 0.10279];
zerO = [0 0 0 0 0 0 0 0 0 0 0 0];
i = 1;

##################################################################################################
#                       Solution
##################################################################################################
global vars = matread("Maximinmatfile.mat");

m = Model(Gurobi.Optimizer);

@variable(m, p[1:G] >= 0 );
@variable(m, RU[1:G] >= 0 );
@variable(m, RD[1:G] >= 0 );
@variable(m, Y[1:G,1:Z] <= 0 );

@constraint(m, GeneratorMaximumCapacity , p + RU .<= pmax);
@constraint(m, GeneratorMinimumCapacity , p - RD .>= pmin);
@constraint(m, MaximumUpwardRegulationCapability , RU .<= Rmax);
@constraint(m, MaximumDownwardRegulationCapability , RD .<= Rmax);
@constraint(m, DAbalance , sum(p) + sum(W*μ') - sum(D) == 0);

@constraint(m, RTbalance[z=1:Z] , sum(Y[:,z]) + W[z,z] == 0);
@constraint(m, RTUpward[g=1:G], sum(Y[g,:]'*Uξ' .- RU[g]) <= 0);
@constraint(m, RTDownward[g=1:G], sum(RD[g] .+ Y[g,:]'*Lξ') >= 0);
@constraint(m, Hypograph, sum(c.*p .+ cU.*RU .+ cD.*RD) .+ sum(c'*Y*Uξ') - 29184.94 <= 0); # k is the constant define in slide nr 17

#sum(c.*vars["XminP"] .+ cU.*vars["XminRU"] .+ cD.*vars["XminRD"]) .+ sum(c'*vars["XminY"]*Uξ')
#15491.891300584395
#sum(c.*vars["XminP"] .+ cU.*vars["XminRU"] .+ cD.*vars["XminRD"]) .+ sum(c'*vars["XminY"]*Lξ')
#29184.939631581386
#sum(c.*vars["XminP"] .+ cU.*vars["XminRU"] .+ cD.*vars["XminRD"]) .+ sum(c'*vars["XminY"]*ξ')
#19236.033542734935

# poly = polyhedron(m, CDDLib.Library(:exact));
# hrepresentation = hrep(m::JuMP.Model);
# colherp = collect(allhalfspaces(hrepresentation));
#function Addition(x,y)
#    return x[:]+y[:];
#end
i = 1;
function Cal_maximal(dataa,dilim)
#    CSV.write("TestCSVPack_Maximin.csv", DataFrame(:x => [dilim[1:2]]), append = i != 0, writeheader = false);
    for ii in [1,2,3,4,5,6,7,8,9,10,11,12]
        zerO = [0 0 0 0 0 0 0 0 0 0 0 0];
        zerO[ii] = 1;
        @objective(m,Max, sum(zerO'.*dataa) ); # this is not an optimisation problem, can we put obj function to be 1?
        optimize!(m);
        Xminobj_p = sum(c.*JuMP.value.(p));
        Xminobj_RU = sum(cU.*JuMP.value.(RU));
        Xminobj_RD = sum(cD.*JuMP.value.(RD));
        Xminobj_YU = sum(sum(c'*JuMP.value.(Y)*Uξ'));
        Xminobj_YL = sum(sum(c'*JuMP.value.(Y)*Lξ'));
        Xminobj_Y = sum(sum(c'*JuMP.value.(Y)*ξ'));
        XminP_norm = norm(JuMP.value.(p) - vars["XminP"],2);
        XminRU_norm = norm(JuMP.value.(RU) - vars["XminRU"],2);
        XminRD_norm = norm(JuMP.value.(RD) - vars["XminRD"],2);
        XminY_norm = norm(JuMP.value.(Y) - vars["XminY"],2);
        Total_norm = norm([XminP_norm,XminRU_norm,XminRD_norm,XminY_norm])/sqrt(4);
        # file = matopen("matfile_dist.mat", "w")
        # write(file, "Xminobj", Xminobj)
        # write(file, "XminP", XminP)
        # write(file, "XminRU", XminRU)
        # write(file, "XminRD", XminRD)
        # write(file, "XminY", XminY)
        # close(file)
        CSV.write("TestCSVPack_Maximin.csv", DataFrame(:x => [Xminobj_p; Xminobj_RU; Xminobj_RD; Xminobj_YU; Xminobj_YL; Xminobj_Y; Total_norm; XminP_norm; XminRU_norm; XminRD_norm]),append = i != 0, writeheader = false);
        CSV.write("TestCSVPack_Maximin.csv", DataFrame(:x => [XminY_norm]), append = i != 0, writeheader = false);
        CSV.write("TestCSVPack_Maximin.csv", DataFrame(:x => ["________________"]),append = i != 0, writeheader = false);
        @objective(m,Min, sum(zerO'.*dataa) ); # this is not an optimisation problem, can we put obj function to be 1?
        optimize!(m);
        Xminobj_p = sum(c.*JuMP.value.(p));
        Xminobj_RU = sum(cU.*JuMP.value.(RU));
        Xminobj_RD = sum(cD.*JuMP.value.(RD));
        Xminobj_YU = sum(sum(c'*JuMP.value.(Y)*Uξ'));
        Xminobj_YL = sum(sum(c'*JuMP.value.(Y)*Lξ'));
        Xminobj_Y = sum(sum(c'*JuMP.value.(Y)*ξ'));
        XminP_norm = norm(JuMP.value.(p) - vars["XminP"],2);
        XminRU_norm = norm(JuMP.value.(RU) - vars["XminRU"],2);
        XminRD_norm = norm(JuMP.value.(RD) - vars["XminRD"],2);
        XminY_norm = norm(JuMP.value.(Y) - vars["XminY"],2);
        Total_norm = norm([XminP_norm,XminRU_norm,XminRD_norm,XminY_norm])/sqrt(4);
        CSV.write("TestCSVPack_Maximin.csv", DataFrame(:x => [Xminobj_p; Xminobj_RU; Xminobj_RD; Xminobj_YU; Xminobj_YL; Xminobj_Y; Total_norm; XminP_norm; XminRU_norm; XminRD_norm]),append = i != 0, writeheader = false);
        CSV.write("TestCSVPack_Maximin.csv", DataFrame(:x => [XminY_norm]), append = i != 0, writeheader = false);
        CSV.write("TestCSVPack_Maximin.csv", DataFrame(:x => ["________________"]),append = i != 0, writeheader = false);
    end
#    CSV.write("TestCSVPack_Maximin.csv", DataFrame(:x => [dilim[1:2]]), append = i != 0, writeheader = false);
    return 1;
end


function Opt_const(OBJECTIVE,dilim,flaG)
#    CSV.write("TestCSVPack_Maximin.csv", DataFrame(:x => [dilim[1:5]]), append = i != 0, writeheader = false);
    if flaG == 1
        for ii in [1,2,3,4,5,6,7,8,9,10,11,12]
            zerO = [0 0 0 0 0 0 0 0 0 0 0 0];
            zerO[ii] = 1;
            @objective(m,Max,sum(zerO'.*OBJECTIVE)); # this is not an optimisation problem, can we put obj function to be 1?
            optimize!(m);
            Xminobj_p = sum(c.*JuMP.value.(p));
            Xminobj_RU = sum(cU.*JuMP.value.(RU));
            Xminobj_RD = sum(cD.*JuMP.value.(RD));
            Xminobj_YU = sum(sum(c'*JuMP.value.(Y)*Uξ'));
            Xminobj_YL = sum(sum(c'*JuMP.value.(Y)*Lξ'));
            Xminobj_Y = sum(sum(c'*JuMP.value.(Y)*ξ'));
            XminP_norm = norm(JuMP.value.(p) - vars["XminP"],2);
            XminRU_norm = norm(JuMP.value.(RU) - vars["XminRU"],2);
            XminRD_norm = norm(JuMP.value.(RD) - vars["XminRD"],2);
            XminY_norm = norm(JuMP.value.(Y) - vars["XminY"],2);
            Total_norm = norm([XminP_norm,XminRU_norm,XminRD_norm,XminY_norm])/sqrt(4);
            CSV.write("TestCSVPack_Maximin.csv", DataFrame(:x => [Xminobj_p; Xminobj_RU; Xminobj_RD; Xminobj_YU; Xminobj_YL; Xminobj_Y; Total_norm; XminP_norm; XminRU_norm; XminRD_norm]),append = i != 0, writeheader = false);
            CSV.write("TestCSVPack_Maximin.csv", DataFrame(:x => [XminY_norm]), append = i != 0, writeheader = false);
            CSV.write("TestCSVPack_Maximin.csv", DataFrame(:x => ["________________"]),append = i != 0, writeheader = false);
            @objective(m,Min,sum(zerO'.*OBJECTIVE)); # this is not an optimisation problem, can we put obj function to be 1?
            optimize!(m);
            Xminobj_p = sum(c.*JuMP.value.(p));
            Xminobj_RU = sum(cU.*JuMP.value.(RU));
            Xminobj_RD = sum(cD.*JuMP.value.(RD));
            Xminobj_YU = sum(sum(c'*JuMP.value.(Y)*Uξ'));
            Xminobj_YL = sum(sum(c'*JuMP.value.(Y)*Lξ'));
            Xminobj_Y = sum(sum(c'*JuMP.value.(Y)*ξ'));
            XminP_norm = norm(JuMP.value.(p) - vars["XminP"],2);
            XminRU_norm = norm(JuMP.value.(RU) - vars["XminRU"],2);
            XminRD_norm = norm(JuMP.value.(RD) - vars["XminRD"],2);
            XminY_norm = norm(JuMP.value.(Y) - vars["XminY"],2);
            Total_norm = norm([XminP_norm,XminRU_norm,XminRD_norm,XminY_norm])/sqrt(4);
            CSV.write("TestCSVPack_Maximin.csv", DataFrame(:x => [Xminobj_p; Xminobj_RU; Xminobj_RD; Xminobj_YU; Xminobj_YL; Xminobj_Y; Total_norm; XminP_norm; XminRU_norm; XminRD_norm]),append = i != 0, writeheader = false);
            CSV.write("TestCSVPack_Maximin.csv", DataFrame(:x => [XminY_norm]), append = i != 0, writeheader = false);
            CSV.write("TestCSVPack_Maximin.csv", DataFrame(:x => ["________________"]),append = i != 0, writeheader = false);
        end
    else
        @objective(m,Max,OBJECTIVE); # this is not an optimisation problem, can we put obj function to be 1?
        optimize!(m);
        Xminobj_p = sum(c.*JuMP.value.(p));
        Xminobj_RU = sum(cU.*JuMP.value.(RU));
        Xminobj_RD = sum(cD.*JuMP.value.(RD));
        Xminobj_YU = sum(sum(c'*JuMP.value.(Y)*Uξ'));
        Xminobj_YL = sum(sum(c'*JuMP.value.(Y)*Lξ'));
        Xminobj_Y = sum(sum(c'*JuMP.value.(Y)*ξ'));
        XminP_norm = norm(JuMP.value.(p) - vars["XminP"],2);
        XminRU_norm = norm(JuMP.value.(RU) - vars["XminRU"],2);
        XminRD_norm = norm(JuMP.value.(RD) - vars["XminRD"],2);
        XminY_norm = norm(JuMP.value.(Y) - vars["XminY"],2);
        Total_norm = norm([XminP_norm,XminRU_norm,XminRD_norm,XminY_norm])/sqrt(4);
        CSV.write("TestCSVPack_Maximin.csv", DataFrame(:x => [Xminobj_p; Xminobj_RU; Xminobj_RD; Xminobj_YU; Xminobj_YL; Xminobj_Y; Total_norm; XminP_norm; XminRU_norm; XminRD_norm]),append = i != 0, writeheader = false);
        CSV.write("TestCSVPack_Maximin.csv", DataFrame(:x => [XminY_norm]), append = i != 0, writeheader = false);
        CSV.write("TestCSVPack_Maximin.csv", DataFrame(:x => ["________________"]),append = i != 0, writeheader = false);
        @objective(m,Min,OBJECTIVE); # this is not an optimisation problem, can we put obj function to be 1?
        optimize!(m);
        Xminobj_p = sum(c.*JuMP.value.(p));
        Xminobj_RU = sum(cU.*JuMP.value.(RU));
        Xminobj_RD = sum(cD.*JuMP.value.(RD));
        Xminobj_YU = sum(sum(c'*JuMP.value.(Y)*Uξ'));
        Xminobj_YL = sum(sum(c'*JuMP.value.(Y)*Lξ'));
        Xminobj_Y = sum(sum(c'*JuMP.value.(Y)*ξ'));
        XminP_norm = norm(JuMP.value.(p) - vars["XminP"],2);
        XminRU_norm = norm(JuMP.value.(RU) - vars["XminRU"],2);
        XminRD_norm = norm(JuMP.value.(RD) - vars["XminRD"],2);
        XminY_norm = norm(JuMP.value.(Y) - vars["XminY"],2);
        Total_norm = norm([XminP_norm,XminRU_norm,XminRD_norm,XminY_norm])/sqrt(4);
        CSV.write("TestCSVPack_Maximin.csv", DataFrame(:x => [Xminobj_p; Xminobj_RU; Xminobj_RD; Xminobj_YU; Xminobj_YL; Xminobj_Y; Total_norm; XminP_norm; XminRU_norm; XminRD_norm]),append = i != 0, writeheader = false);
        CSV.write("TestCSVPack_Maximin.csv", DataFrame(:x => [XminY_norm]), append = i != 0, writeheader = false);
        CSV.write("TestCSVPack_Maximin.csv", DataFrame(:x => ["________________"]),append = i != 0, writeheader = false);
    end
#    CSV.write("TestCSVPack_Maximin.csv", DataFrame(:x => [dilim[1:5]]), append = i != 0, writeheader = false);
    return 1;
end

Cal_maximal(p,"p ");
Cal_maximal(RU,"RU");
Cal_maximal(RD,"RD");
for j in [1,2,3,4]
    Cal_maximal(Y[:,j],"Y ");
end
Opt_const(p + RU,"obj1 ",1)
Opt_const(p - RD,"obj2 ",1)
Opt_const(sum(p) + sum(W*μ') - sum(D),"obj3 ",0)
Opt_const(sum(Y[:,1:Z]) + sum(W[1:Z,1:Z]),"obj4 ",0)
Opt_const(sum(Y[1:G,:]*Uξ' .- RU[1:G]),"obj5 ",0)
Opt_const(sum(RD[1:G] .+ Y[1:G,:]*Lξ'),"obj6 ",0)
Opt_const(sum(c.*p .+ cU.*RU .+ cD.*RD) .+ sum(c'*Y*Uξ'),"obj7 ",0)

Opt_const(sum(p - RU),"obj1_3",3)
Opt_const(sum(p + RD),"obj2_3",3)
Opt_const(sum(p) - sum(W*μ') - sum(D),"obj3_4",4)
Opt_const(sum(Y[:,1:Z]) - sum(W[1:Z,1:Z]),"obj4_4",4)
Opt_const(sum(Y[1:G,:]*Uξ' .+ RU[1:G]),"obj5_4",4)
Opt_const(sum(RD[1:G] .- Y[1:G,:]*Lξ'),"obj6_4",4)
Opt_const(sum(c.*p .- cU.*RU .+ cD.*RD) .+ sum(c'*Y*Uξ'),"obj7_4",4)

Opt_const(sum(p) + sum(W*μ') + sum(D),"obj3_4",4)
Opt_const(sum(Y[:,1:Z]),"obj4_4",4)
Opt_const(sum(sum(W[1:Z,1:Z])),"obj4_4",4)
Opt_const(sum(Y[1:G,:]*Uξ'),"obj5_4",4)
Opt_const(sum(RU[1:G]),"obj5_4",4)
Opt_const(sum(RD[1:G]),"obj6_4",4)
Opt_const(sum(Y[1:G,:]*Lξ'),"obj6_4",4)
Opt_const(sum(c.*p .+ cU.*RU .- cD.*RD) .+ sum(c'*Y*Uξ'),"obj7_4",4)
Opt_const(sum(c.*p .+ cU.*RU .+ cD.*RD) .- sum(c'*Y*Uξ'),"obj7_4",4)
Opt_const(sum(c.*p .- cU.*RU .- cD.*RD) .+ sum(c'*Y*Uξ'),"obj7_4",4)
Opt_const(sum(c.*p .+ cU.*RU .- cD.*RD) .- sum(c'*Y*Uξ'),"obj7_4",4)
Opt_const(sum(c.*p .- cU.*RU .+ cD.*RD) .- sum(c'*Y*Uξ'),"obj7_4",4)
Opt_const(sum(c.*p .- cU.*RU .- cD.*RD) .- sum(c'*Y*Uξ'),"obj7_4",4)
Opt_const(sum(-c.*p .- cU.*RU .- cD.*RD) .+ sum(c'*Y*Uξ'),"obj7_4",4)
Opt_const(sum(-c.*p .+ cU.*RU .- cD.*RD) .- sum(c'*Y*Uξ'),"obj7_4",4)
Opt_const(sum(-c.*p .- cU.*RU .+ cD.*RD) .- sum(c'*Y*Uξ'),"obj7_4",4)
Opt_const(sum(-c.*p .+ cU.*RU .- cD.*RD) .+ sum(c'*Y*Uξ'),"obj7_4",4)
Opt_const(sum(-c.*p .- cU.*RU .+ cD.*RD) .+ sum(c'*Y*Uξ'),"obj7_4",4)
Opt_const(sum(-c.*p .+ cU.*RU .+ cD.*RD) .- sum(c'*Y*Uξ'),"obj7_4",4)

#print(m)
#optimize!(m)

#println("Objective value: ", JuMP.objective_value(m))
#println("p=", JuMP.value.(p))
#println("RU=", JuMP.value.(RU))
#println("RD=", JuMP.value.(RD))
#println("Y=", JuMP.value.(Y))
