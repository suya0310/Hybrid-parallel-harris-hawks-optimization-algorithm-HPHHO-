% Main paper:
% A Hybrid Parallel Harris Hawks Optimization Algorithm for Reusable Launch Vehicle Reentry Trajectory Optimization with No-Fly Zones
% Su, Ya, Dai, Ying, and Liu, Yi
% Soft Computing, Vol. 25, No. 23, 2021, pp. 14597–14617.
% https://doi.org/10.1007/s00500-021-06039-y.

% I acknowledge that this version of HPHHO has been written using
% a large portion of the following code:
% Harris hawks optimization: Algorithm and applications
% Ali Asghar Heidari, Seyedali Mirjalili, Hossam Faris, Ibrahim Aljarah, Majdi Mafarja, Huiling Chen
% Future Generation Computer Systems, 
% DOI: https://doi.org/10.1016/j.future.2019.02.028

function [Rabbit_Energy,Rabbit_Location,CNVG,Total_time]=HPHHO(N,T,lb,ub,dim,fitness)
disp('HPHHO is now tackling your problem')
tic
% initialize the location and Energy of the rabbit
Rabbit_Location=zeros(1,dim);
Rabbit_Energy=1e20;
CP=ceil(N/2);
X_HHO=zeros(CP,dim);
X_DE=zeros(CP,dim);
Xnew=zeros(CP,dim);
CNVG=zeros(1,T);
%Initialize the locations of Harris' hawks
X=initialization(N,dim,ub,lb);
Tfitness=zeros(2*N,1);
w=4;
r=rand;
while (r==0.25||r==0.5||r==0.75)
    r=rand;
end
F=0.5;      % scaling factor
CR=0.9;     % crossover rate
t=0;        % Loop counter

while t<T  

     r=w*r*(1-r);
     OBLPositions=OBLinitialization(N,dim,lb,ub,X);  
   % Return back the search agents that go beyond the boundaries of the search space
   for i=1:size(OBLPositions,1) 
      Flag4ub=OBLPositions(i,:)>ub;      Flag4lb=OBLPositions(i,:)<lb;
      OBLPositions(i,:)=(OBLPositions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb; 
   end
   
    TPositions=vertcat(X,OBLPositions);
   % the fitness of the total solutions 
    for i=1:2*N        
        Tfitness(i,1)=fitness(TPositions(i,:));          
    end 
    [Val,Tindex]=sort(Tfitness);             %求极小值 从小到大排序      
    for newindex=1:N
        X(newindex,:)=TPositions(Tindex(newindex),:);       
    end  
    
    TestRabbit_Energy=Val(1);      
    if TestRabbit_Energy<Rabbit_Energy
       Rabbit_Energy=TestRabbit_Energy;
       Rabbit_Location=X((1),:);
    end 
       
    for i=1:CP
        X_DE(i,:)=X(i,:);
        X_HHO(i,:)=X(CP+i,:);
    end    
%%************************ smoothing technique *******************************************
     for i=1:CP
         for j=2:dim-1
             if ((X_HHO(i,j)-X_HHO(i,j-1))*(X_HHO(i,j+1)-X_HHO(i,j)))<0
                 X_HHO(i,j)=0.5*(X_HHO(i,j-1)+X_HHO(i,j+1));
             end
             if ((X_DE(i,j)-X_DE(i,j-1))*(X_DE(i,j+1)-X_DE(i,j)))<0
                 X_DE(i,j)=0.5*(X_DE(i,j-1)+X_DE(i,j+1));
             end
         end
          Flag4ub=X_DE(i,:)>ub; Flag4lb=X_DE(i,:)<lb;
          X_DE(i,:)=(X_DE(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;  
     end 
    %******************************************  HHO  ****************************************************
    E1=2*(1-(t/T)); % factor to show the decreaing energy of rabbit
    % Update the location of Harris' hawks
    for i=1:CP
        E0=2*rand()-1; %-1<E0<1
%         E0=2*p-1; %-1<E0<1
        Escaping_Energy=E1*(E0);  % escaping energy of rabbit     
        
        if abs(Escaping_Energy)>=1
            %% Exploration:
            % Harris' hawks perch randomly based on 2 strategy:            
            q=rand();   
            kk=randperm(CP);
            kk(i==kk)=[];          
            if q<0.5
                % perch based on other family members
%                X_HHO(i,:)=Rabbit_Location+F*(X_HHO(kk(1),:)-X_HHO(kk(2),:));
%                 X(i,:)=X_rand-rand()*abs(X_rand-2*rand()*X(i,:));
                 X_HHO(i,:)=X_HHO(i,:)+F*(Rabbit_Location-X_HHO(kk(1),:)+X_HHO(kk(2),:)-X_HHO(kk(3),:));
            elseif q>=0.5
                % perch on a random tall tree (random site inside group's home range)
                  X_HHO(i,:)=(Rabbit_Location(1,:)-mean(X_HHO))-rand()*((ub-lb)*rand+lb);
            end               
                  Flag4ub=X_HHO(i,:)>ub;   Flag4lb=X_HHO(i,:)<lb;
                  X_HHO(i,:)=(X_HHO(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb; 
        elseif abs(Escaping_Energy)<1
            %% Exploitation:
            % Attacking the rabbit using 4 strategies regarding the behavior of the rabbit            
            %% phase 1: surprise pounce (seven kills)
            % surprise pounce (seven kills): multiple, short rapid dives by different hawks            
%             r=rand(); % probablity of each event   %  !!!!!!!!!!!!!!!!!(original rand)           
            if r>=0.5 && abs(Escaping_Energy)<0.5 % Hard besiege
                X_HHO(i,:)=(Rabbit_Location)-Escaping_Energy*abs(Rabbit_Location-X_HHO(i,:));
            end
            
            if r>=0.5 && abs(Escaping_Energy)>=0.5  % Soft besiege
                Jump_strength=2*(1-rand()); % random jump strength of the rabbit
                X_HHO(i,:)=(Rabbit_Location-X_HHO(i,:))-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X_HHO(i,:));
            end   
            
               Flag4ub=X_HHO(i,:)>ub; Flag4lb=X_HHO(i,:)<lb;
               X_HHO(i,:)=(X_HHO(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb; 
            
            %% phase 2: performing team rapid dives (leapfrog movements)
            if r<0.5 && abs(Escaping_Energy)>=0.5   % Soft besiege % rabbit try to escape by many zigzag deceptive motions
                
                Jump_strength=2*(1-rand());
                X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X_HHO(i,:));
                
                Flag4ub=X1>ub; Flag4lb=X1<lb; X1=(X1.*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;     
                
                if fitness(X1)<fitness(X_HHO(i,:))    % improved move?
                    X_HHO(i,:)=X1;
                else % hawks perform levy-based short rapid dives around the rabbit
                    X2=(Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X_HHO(i,:))).*(ones(1,dim)+Levy(dim));
%                     X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X_HHO(i,:))+rand(1,dim).*Levy(dim);    
                    Flag4ub=X2>ub; Flag4lb=X2<lb; X2=(X2.*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;                     
                    if (fitness(X2)<fitness(X_HHO(i,:)))  % improved move?
                        X_HHO(i,:)=X2;
                    end
                end
            end
            
            if r<0.5 && abs(Escaping_Energy)<0.5  % Hard besiege % rabbit try to escape by many zigzag deceptive motions
                % hawks try to decrease their average location with the rabbit
                Jump_strength=2*(1-rand());
                X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-mean(X_HHO));
                
                Flag4ub=X1>ub;Flag4lb=X1<lb; X1=(X1.*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;                   
                if fitness(X1)<fitness(X_HHO(i,:)) % improved move?
                    X_HHO(i,:)=X1;
                else % Perform levy-based short rapid dives around the rabbit
                    X2=(Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-mean(X_HHO))).*(ones(1,dim)+Levy(dim));
%                      X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-mean(X_HHO))+rand(1,dim).*Levy(dim);                 
                    Flag4ub=X2>ub; Flag4lb=X2<lb; X2=(X2.*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;                     
                    if (fitness(X2)<fitness(X_HHO(i,:)))   % improved move?
                        X_HHO(i,:)=X2;
                    end
                end
            end
            %%
        end  
    %******************************** DE *****************************************************         
        %/*Randomly selected solution must be different from the solution i*/        
            kkk=randperm(CP);
            kkk(i==kkk)=[];
            jrand=randi(dim);  
            for j=1:dim        
                  if (rand<=CR)||(jrand==j)
%                    Xnew(i,j)=X_DE(i,j)+F*(Rabbit_Location(j)-X_DE(kkk(1),j)+X_DE(kkk(2),j)-X_DE(kkk(3),j));
                    Xnew(i,j)=Rabbit_Location(j)+F*(X_DE(kkk(1),j)-X_DE(kkk(2),j));
                  else
                    Xnew(i,j)=X_DE(i,j);             
                  end
            end           
            % Return back the search agents that go beyond the boundaries of the search space    
             Flag4ub=Xnew(i,:)>ub; Flag4lb=Xnew(i,:)<lb;             
             Xnew(i,:)=(Xnew(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;        
             if fitness(Xnew(i,:))<fitness(X_DE(i,:))
                   X_DE(i,:)=Xnew(i,:);                       
             end               
    end     
      X=vertcat(X_DE,X_HHO);
    %******************************* SELECT ***********************************
    for i=1:N       
        Flag4ub=X(i,:)>ub;        Flag4lb=X(i,:)<lb;
        X(i,:)=(X(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;  
        Fit=fitness(X(i,:));        
              if Fit<Rabbit_Energy
                   Rabbit_Energy=Fit;
                   Rabbit_Location=X(i,:);
              end          
    end
    t=t+1;
    CNVG(t)=Rabbit_Energy;
    display(['Iteration:',num2str(t), '   Leader_score:',num2str(Rabbit_Energy)]);   
end
Total_time=toc;
end
% ___________________________________
function o=Levy(d)
beta=1.5;
sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
u=randn(1,d)*sigma;v=randn(1,d);step=u./abs(v).^(1/beta);
o=step;
end



