import pandas as pd 
import numpy as np 
import os, sys
from geopandas import GeoDataFrame, GeoSeries
import fiona
from shapely.geometry import Point, Polygon
from geopandas.tools import sjoin
from datetime import datetime
import re
from rasterstats import zonal_stats

os.chdir("/Users/jozo/Dropbox/_Projects/ubc-micromet/DIYSCO2-main/")
# sys.path.insert(0, os.getcwd()+'/mapping/libs/')
# import grid as g


# from filterRaw import filterRaw
# from makeGeo import makePoints, makeGrid
# from gridGroupStats import calcgeostats
# from gridEmissions import gE
''' --- Global Functions --- '''
# Match datetimeindex - assumes we read in datetime as index column
def matchtimeindex(data, start, end):
    timevalues = pd.date_range(start, end, periods=None, freq='S')
    output = data[data.index.isin(timevalues)]
    return output

def readdata(ipath):
    output = pd.read_csv(ipath, header = 0, index_col='datetime', parse_dates=True)
    return output

def featcount(datalist):
    output=[len(i) for i in datalist]
    print output

def labDataFilter(df):
    '''
    # filter data in the lab - excludes the speed component
    '''
    df = df[(df.co2 >380) & (df.co2 < 1000) & (df.tcell>45) & (df.pcell > 96)]
    return df


def dropDuplicatesByLast(df):
    '''
    # drop duplicates by the last feature
    '''
    df = df.groupby(df.index).last()
    return df

def joinResampledList(datalist, start, end, variable, timeFreq):
    '''
    # join all the co2 to a table
    '''
    time = pd.Series(pd.date_range(start, end, freq=timeFreq), name="datetime"); 
    output = pd.DataFrame(index=time); #print test
    # create empty array to append to
    cnames=[]
    for i in range(len(datalist)):
        # get the sensor id of each df
        sensorid = str("co2_"+str(datalist[i]['sensorid'].ix[0]))
        # append the sensor id name to the empty array
        cnames.append(sensorid)
        # get the co2 field from each df in the list and append this as a column
        output = output.join(datalist[i]['co2'], lsuffix = "_"+sensorid)
    # set the column names as cnames
    output.columns = cnames
    # calculate the mean across the row
    output['meanco2'] = output.mean(axis=1)
    print output.head()
    return output

def calcDrift(df, colname):
    name = colname + "_drift"
    df[name] = df[colname] - df['meanco2']
    return df


def getCalibrationValues(path, start, end, whichOne, experiment):

    # read in data to array
    ipaths = [os.path.join(path,i) for i in os.listdir(path) if i.endswith('.csv')]
    data = [readdata(i) for i in ipaths]
    # print number of measurements
    featcount(data)

    # Specify column types as float
    for i in range(0,len(data)):
        data[i].co2 = data[i].co2.astype('float')
        data[i].pcell = data[i].pcell.astype('float')
        data[i].tcell = data[i].tcell.astype('float')
        data[i].tempin = data[i].tempin.astype('float')
        data[i].tempout = data[i].tempout.astype('float')   

    # --- Filter data ---#
    data = [labDataFilter(i) for i in data]
    featcount(data)

    # drop duplicates
    data = [dropDuplicatesByLast(i) for i in data]
    featcount(data)

    # --- bin by time --- #
    data_1min = [i.resample('1min', how={'co2': np.mean, 'pcell': np.mean, 'tcell':np.mean, 'tempin':np.mean, 'tempout':np.mean, 'sensorid':np.mean}) for i in data]
    # replace inf
    data_1min = [i.replace([np.inf, -np.inf], np.nan) for i in data_1min]
    

    # print(data_1min)

    ''' --- output ---- '''
    # --- create a df using join var for each time stamp & calc the mean --- #
    # for each second
    co2main = joinResampledList(data, start, end, 'co2', "S")

    if whichOne == "initial":
        dbpath = os.getcwd() + '/diysco2-db/campaigns/'+experiment+'/insitu-calibration-start/'
    elif whichOne == "end":
        dbpath = os.getcwd() + '/diysco2-db/campaigns/'+experiment+'/insitu-calibration-end/'

    print dbpath
    for i in co2main.columns:
        co2main = calcDrift(co2main, i)
    co2main.to_csv(dbpath+"/data-merged-1sec.csv")

    # resample for each minute
    co2main_1min = joinResampledList(data_1min, start, end, "co2", "1min")

    for i in co2main_1min.columns:
        co2main_1min = calcDrift(co2main_1min, i)
    co2main_1min.to_csv(dbpath+"/data-merged-1min.csv")

    # return the calibration values
    return {
        'calibration_1sec': co2main,
        'calibration_1min': co2main_1min
    }


def filterRaw(path, start, end, experiment):
    # read in data to array
    ipaths = [os.path.join(path,i) for i in os.listdir(path) if i.endswith('.csv')]

    # experiment data
    data = [readdata(i) for i in ipaths]
    # --- print number of measurements --- #
    featcount(data)

    # --- match datetime --- #
    # drop duplicates
    data = [i.groupby(i.index).last() for i in data]
    featcount(data)
    # data[1].drop(pd.Timestamp('2015-01-01 01:00:00'))

    # --- Subset data where datetimes match for a given period --- #
    data = [ matchtimeindex(i, start, end) for i in data]
    featcount(data)

    # --- Offset by total delayed response time 18 seconds // 16s from inlet to irga, 3.2 within irga --- #
    # make data copy called ldata
    ldata = [i.copy() for i in data]
    # df.gdp = df.gdp.shift(-1)
    for i in ldata:
        i.co2 = i.co2.shift(-18)

    # adjust for the sensor drift - determiend from the calibration analysis
    for i in ldata:
        if experiment == '160318':
            if i.sensorid.iloc[0] == '0108':
                i.co2 = i.co2 + 1.24
            elif i.sensorid.iloc[0] == '0205':
                i.co2 = i.co2 - 5.84         
            elif i.sensorid.iloc[0] == '0150':
                i.co2 = i.co2 - 7.75            
            elif i.sensorid.iloc[0] == '0151':
                i.co2 = i.co2 + 1.23       
            elif i.sensorid.iloc[0] == '1641':
                i.co2 = i.co2 - 0.58      
        elif experiment == '150318':
            if i.sensorid.iloc[0] == '0108':
                i.co2 = i.co2 + 0.37
            elif i.sensorid.iloc[0] == '0205':
                i.co2 = i.co2 - 3.29
            elif i.sensorid.iloc[0] == '0150':
                i.co2 = i.co2 - 1.31
            elif i.sensorid.iloc[0] == '0151':
                i.co2 = i.co2 + 4.69      
            elif i.sensorid.iloc[0] == '1641':
                i.co2 = i.co2 - 0.46   

    # --- filter data --- #
    '''
    # only return data if:
    1. co2 ppm between 380 & 1000ppm, 
    2. cell temperature is greater than 45C
    3. cell pressure is greater than 96
    4. speed less than 5kmh
    '''
    ldata = [ i[(i.co2 >380) & (i.co2 < 1000) & (i.tcell>45) & (i.pcell > 96) & (pd.isnull(i.lat) ==False) ] for i in ldata]

    ''' ------------- fix temp out for 0108 & 0150 ------ '''
    # 0108
    ldata[0]['tempin1'] = ldata[0]['tempout']
    ldata[0]['tempout'] = ldata[0]['tempin']
    ldata[0]['tempin'] = ldata[0]['tempin1']
    ldata[0].drop('tempin1', axis=1, inplace=True)

    # 0150
    ldata[1]['tempin1'] = ldata[1]['tempout']
    ldata[1]['tempout'] = ldata[1]['tempin']
    ldata[1]['tempin'] = ldata[1]['tempin1']
    ldata[1].drop('tempin1', axis=1, inplace=True)

    # filter for speeds greater than 5kph 
    lts = 5
    ldata = [ i[(i.speed >lts)] for i in ldata]
    featcount(ldata)  

    ''' ----------- Export cleaned/filtered data ------- '''
    
    
    opath = os.getcwd() + '/diysco2-db/campaigns/'+experiment+'/diysco2-filtered/'
    if(os.path.isdir(opath)):
        print "already a folder!"
    else:
        os.mkdir(opath)

    for i in ldata:
        oname = str("sensor_"+str(i.sensorid.ix[0])+"_"+experiment+".csv"); print oname
        i.to_csv(opath+"/"+oname)

    # return the datalist
    return ldata


def makePoints(experiment):
    path = os.getcwd() + '/diysco2-db/campaigns/'+experiment+'/diysco2-filtered/'
    ipaths = [os.path.join(path,i) for i in os.listdir(path) if i.endswith('.csv')]
    # experiment data
    data = [readdata(i) for i in ipaths]
    # copy data to ldata 
    ldata = [i.copy() for i in data]
    # --- print number of measurements --- #
    featcount(ldata)

    ''' ------------- Spatial Operations ------------- '''
    for i in range(0,len(ldata)):
        ldata[i].lon = ldata[i].lon.astype('float')
        ldata[i].lat = ldata[i].lat.astype('float')
        
    # need to keep datetime field
    # create geopoints
    for i in ldata:
        i['datetime'] = i.index
        
        
    for i in ldata:
        i.index = [j for j in range(len(i))]    
        i['geometry'] = GeoSeries([Point(x, y) for x, y in zip(i.lon, i.lat)])
        # convert datetime string to iso format
        i['datetime'] = i.datetime.map(lambda x: datetime.strftime(x, '%Y-%m-%dT%H:%M:%SZ'))

    print ldata[0].head()
    # Projections 
    gridproj = {'init': 'epsg:3740', 'no_defs': True}
    wgs84 = {'datum':'WGS84', 'no_defs':True, 'proj':'longlat'}

    # create geodataframe from data
    ldata = [GeoDataFrame(i) for i in ldata]

    # set projection as wgs84
    for i in ldata:
        i.crs = wgs84

    # reproject to utm zone 10N
    for i in ldata: 
        i.geometry = i.geometry.to_crs(epsg=3740)
        # i.geometry = i.geometry.to_crs(epsg=4326)

    for i in ldata:
        i = i[pd.isnull(i.geometry) == False]

    # --- Merge geodata together --- #
    mergedgeo = pd.concat([ldata[0], ldata[1],ldata[2],ldata[3],ldata[4]])
    mergedgeo = GeoDataFrame(mergedgeo)
    mergedgeo.crs = gridproj

    print len(mergedgeo)
    mergedgeo = mergedgeo[pd.isnull(mergedgeo.lat)==False]
    print len(mergedgeo)
    # mergedgeo['date'] = mergedgeo['date'].str.replace('/', '-').astype(str)
    # mergedgeo['datetime'] = mergedgeo['datetime'].astype(str)

    print mergedgeo.head()
    # mergedgeo.to_crs(wgs84)

    opath = os.getcwd() + '/diysco2-db/campaigns/'+experiment+'/diysco2-filtered-points/'
    
    print opath
    if(os.path.isdir(opath)):
        print "already a folder!"
    else:
        os.mkdir(opath)

    if(os.path.isfile(opath + 'all_20150528.geojson')):
        os.remove(opath + 'all_20150528.geojson')

    mergedgeo.to_file(opath + 'all_20150528.geojson', driver="GeoJSON")
    # with open(opath + 'all_20150528.geojson', 'w') as f:
    #       f.write(mergedgeo.to_json())
    mergedgeo.to_file(opath + 'all_20150528.shp', driver='ESRI Shapefile')
    
    return mergedgeo
    del mergedgeo

def makeGrid(ipoints, experiment, gridsize):
    # Projections 
    gridproj = {'init': 'epsg:3740', 'no_defs': True}
    wgs84 = {'datum':'WGS84', 'no_defs':True, 'proj':'longlat'}
    # import grid script
    sys.path.insert(0, os.getcwd()+'/mapping/libs/')
    import grid as g

    opath =  os.getcwd() + '/diysco2-db/campaigns/'+experiment+'/diysco2-grid'
    if(os.path.isdir(opath)):
        print "already a folder!"
    else:
        os.mkdir(opath)

    # gridsize = 200
    ogridname = "grid_"+str(gridsize)+"m.shp"
    ofile = opath + "/" + ogridname
    print "making grid"
    g.main(ofile, ipoints.total_bounds[0], ipoints.total_bounds[2], 
        ipoints.total_bounds[1], ipoints.total_bounds[3],
        gridsize, gridsize)

    print "grid complete! "
    # read in the grid that was just made
    grid = GeoDataFrame.from_file(ofile)
    grid.crs = gridproj
    # create grid id to groupby
    grid['id'] = [i for i in range(len(grid))]

    # Read in transect to spatial subset grids in transect
    transect = GeoDataFrame.from_file(os.getcwd()+'/diysco2-db/_main_/study-area/' +'transect_epicc2sp_woss.shp')
    transect.crs = gridproj

    # subset grid
    # transectgrid = grid[grid.geometry.intersects(transect.geometry)]; print transectgrid
    sagrid = []
    for i in range(len(grid)):
        if np.array(transect.intersects(grid.geometry[i]))[0] != False:
            sagrid.append(grid.geometry[i])

    transectgrid = GeoDataFrame(sagrid)
    transectgrid.columns = ['geometry']
    transectgrid['id'] = [i for i in range(len(transectgrid))]
    transectgrid.crs = gridproj

    

    transectgrid.to_file(ofile[:-4]+"_transect.shp")
    # transectgrid.to_file(ofile[:-4]+"_transect.geojson",driver="GeoJSON")

    ## !!!Some weird things with reading in data makes the sjoin work !!! :(
    transectgrid = GeoDataFrame.from_file(ofile[:-4]+"_transect.shp")
    transectgrid.crs = gridproj
    print transectgrid.head()

    ipoints = GeoDataFrame.from_file( os.getcwd() + '/diysco2-db/campaigns/'+experiment+'/diysco2-filtered-points/all_20150528.shp')
    ipoints.crs = gridproj
    print ipoints.head()

    # ipoints['id'] = [i for i in range(len(ipoints))]
    # Spatial join points to grid
    oname = "gridjoin_"+str(gridsize)+"m.shp"
    # join_inner_df = sjoin(transectgrid, ipoints, how="inner")
    join_inner_df = sjoin(transectgrid, ipoints, how="left", op='intersects')
    # join_inner_df.to_file(opath+ "/"+oname)

    return join_inner_df

def describestats(data, variable):
    output = pd.DataFrame()
    output[str(variable + "_cnt")] = data.groupby(['id'])[variable].count()
    output[str(variable + "_avg")] = data.groupby(['id'])[variable].mean()
    output[str(variable + "_med")] = data.groupby(['id'])[variable].median()
    output[str(variable + "_skew")] = data.groupby(['id'])[variable].skew()
    output[str(variable + "_var")] = data.groupby(['id'])[variable].var()
    output[str(variable + "_stdev")] = data.groupby(['id'])[variable].std()
    output[str(variable + "_min")] = data.groupby(['id'])[variable].min()
    output[str(variable + "_max")] = data.groupby(['id'])[variable].max()
    output[str(variable + "_rng")] = data.groupby(['id'])[variable].max() - data.groupby(['id'])[variable].min()
    return output


def calcgeostats(gridjoin, experiment, gridsize):

    # read in merged gridded geodata 
    # join_inner_df = GeoDataFrame.from_file('/Users/Jozo/Dropbox/_Projects/_GitHub/MobileCO2/Projects/05_GroupTraverse_01/data/filtered_geojson_wgs84/output/gridjoin_250m.shp')
    # join_inner_df = GeoDataFrame.from_file(iofolder+'diysco2-grid/'+ 'gridjoin_'+str(gridsize)+'m'+'.shp')
    join_inner_df = gridjoin
    # read in grid
    # path2grid = "/Users/Jozo/Dropbox/_Projects/_GitHub/MobileCO2/Projects/05_GroupTraverse_01/data/grid/grid_250m_transect.shp"
    # path2grid = iofolder+'diysco2-grid/'+"grid_"+str(gridsize)+'m'+"_transect.shp"
    path2grid = os.getcwd() + '/diysco2-db/campaigns/'+experiment+'/diysco2-grid/'+'grid_'+str(gridsize)+'m'+"_transect.shp"
    
    grid = GeoDataFrame.from_file(path2grid)
    # create grid id to groupby
    grid['id'] = [i for i in range(len(grid))]

    # run describestats function 
    co2_stats = describestats(join_inner_df, 'co2')
    temp_stats = describestats(join_inner_df, 'tempout')
    # Merge data together
    summarystats = pd.merge(co2_stats, temp_stats, left_index=True, right_index=True)

    # merge data to geogrid
    output = GeoDataFrame.merge(grid, summarystats, left_on="id", right_index=True)
    output = output.reset_index()
    # output = output.fillna(-9999)

    opath = os.getcwd() + '/diysco2-db/campaigns/'+experiment+'/diysco2-grid/'
    if(os.path.isdir(opath)):
        print "already a folder!"
    else:
        os.mkdir(opath)

    # output.to_file('/Users/Jozo/Dropbox/_Projects/_GitHub/MobileCO2/Projects/05_GroupTraverse_01/data/filtered_geojson_wgs84/output/gridstats_250m.shp')
    # output.to_file(opath+'/'+'gridstats_'+str(gridsize)+'m'+'.shp')
    output.to_file(os.getcwd() + '/diysco2-db/campaigns/'+experiment+'/diysco2-grid/'+'gridstats_'+str(gridsize)+'m'+".shp")
    # output.to_file(opath+'/'+'gridstats_'+str(gridsize)+'m'+'.geojson', driver="GeoJSON")
    return output






def joinTrafficCounts(data_grid, utm10n):
    data_grid['gid'] = data_grid.id
    data_grid.crs = utm10n; print data_grid.crs

    osm_trafficCounts_centroids = GeoDataFrame.from_file(os.getcwd()+'/diysco2-db/_main_/yvr-open-data-traffic-counts/generated-traffic-counts-osm-split/'+'osm_trafficCounts_split_dev_'+str(50)+'_centroids.shp')
    gridded_traffic_counts = sjoin(data_grid, osm_trafficCounts_centroids, how="left")
    print len(gridded_traffic_counts)

    return gridded_traffic_counts 


def generateDtTrafficArray(traffic_hour_array):
    dt_traffic_hour_array = []
    for i in traffic_hour_array:
        dt_name = re.sub('tc', 'dt',i)
        dt_traffic_hour_array.append(dt_name)

    return dt_traffic_hour_array


''' --- calculate emission per grid cell --- '''
def getDistanceTravelled(osm_trafficCounts_all, traffic_hour_array, dt_traffic_hour_array):
    for i in range(len(traffic_hour_array)):
        osm_trafficCounts_all[dt_traffic_hour_array[i]] = osm_trafficCounts_all[traffic_hour_array[i]]*osm_trafficCounts_all['len_m']

    return osm_trafficCounts_all

''' --- calculate emission per grid cell --- '''
def sumByGridId(osm_trafficCounts_all, traffic_hour_array, dt_traffic_hour_array):
    output = pd.DataFrame()
    for i in dt_traffic_hour_array+traffic_hour_array +["len_m"]:
        output[i] = osm_trafficCounts_all.groupby(['gid'])[i].sum()

    return output

def mergeTrafficCountsToGrid(data_grid, osm_trafficCounts_all):
    data_grid['gid'] = data_grid['id']
    gridded_data = GeoDataFrame.merge(data_grid, osm_trafficCounts_all, left_on="gid", right_index=True, how="left")
    return gridded_data

def generateETrafficArray(traffic_hour_array):
    e_traffic_hour_array = []
    for i in traffic_hour_array:
        e_name = re.sub('tc', 'e',i)
        e_traffic_hour_array.append(e_name)

    return e_traffic_hour_array


def calculateTrafficEmissions(gridded_data, monthFactor, emissionsFactor, vehicleEfficiency, dt_traffic_hour_array, e_traffic_hour_array):
    for i in range(len(dt_traffic_hour_array)):
        # gridded_data[i] is in meters
        # kg CO2 per meter driven in the grid cell
        # gridded_data[e_name] = (((gridded_data[i]* vehicleEfficiency) * emissionsFactor) * monthFactor) / gridScalingFactor
        gridded_data[e_traffic_hour_array[i]] = (((gridded_data[dt_traffic_hour_array[i]]* vehicleEfficiency) * emissionsFactor) * monthFactor)   

    return gridded_data

def roundVals(gridded_data, dt_traffic_hour_array, e_traffic_hour_array, gridsize):
    # round vals to 3 sig figures
    for i in dt_traffic_hour_array:
        gridded_data[i] = gridded_data[i].round(3)
    # for i in fb_traffic_hour_array:
    #   gridded_data[i] = gridded_data[i].round(3)
    for i in e_traffic_hour_array:
        if gridsize == 50:
            gridded_data[i] = gridded_data[i].round(15) 
        else:
            gridded_data[i] = gridded_data[i].round(3)  

    return gridded_data

def calcPeriodHourly(gridded_data, variable, elapsedtime):
    gridded_data[variable+"_hr"] = gridded_data[variable] / (elapsedtime) # kg CO2 ha-1 hr-1

    return gridded_data

def calcMeasuredEmissions(data, densityOfAir, molarDensityOfCo2, molarMassOfAir, backgroundCo2Concentrations, rah, gridsize):
    gridsize2sqm = gridsize*gridsize # to get per hectare regardless of the grid size
    mobileco2concentrations = data['co2_avg'] * densityOfAir / molarMassOfAir 
    mobileco2concentrations = mobileco2concentrations *  molarDensityOfCo2 / 1000.0 # g to kg

    background = backgroundCo2Concentrations * densityOfAir / molarMassOfAir 
    background = background * molarDensityOfCo2/1000.0
    # modelled emissions
    data['co2_avg_e'] = (mobileco2concentrations - background)/ rah # g CO2 m-2 s-1
    # convert to kg CO2 ha-1 hr-1
    data['co2_avg_e'] = data['co2_avg_e']*3600.0 # g co2 m-2 hr-1
    data['co2_avg_e'] = data['co2_avg_e'] / 1000.0 # kg co2 m-2 hr-1
    data['co2_avg_e'] = data['co2_avg_e'] * gridsize2sqm # kg CO2 per grid area hr-1
    # data['co2_avg_e'] = data['co2_avg_e'] / (gridsize2sqm / 10000.0) # kg co2 ha-1 hr-1

    print ("the max is: ", data['co2_avg_e'].max())
    print ("the mean is: ", data['co2_avg_e'].mean())

    return data

def addNeighborhoods(data, utm10n):
    hoods = GeoDataFrame.from_file(os.getcwd()+'/diysco2-db/_main_/yvr-open-data-neighborhoods/csg_neighborhood_areas.shp'); print hoods.crs
    hoods.crs = utm10n
    output = data.copy()
    output.iscopy = False
    print len(output)
    output = sjoin(output, hoods, how="left")
    output['temp'] = [str(i.bounds) for i in output.geometry]
    print output['temp'].head()
    output = output.drop_duplicates('temp', keep="last")
    print len(output)

    # output.index = [i for i in range(len(otu))]
    for i in range(len(output)):
        if output['NAME'].iloc[i] is None:
            output['NAME'].iloc[i] = "Stanley Park"

        if output["MAPID"].iloc[i] is None:
            output['MAPID'].iloc[i] = "SP1"

    # output = output[pd.isnull(output.co2_avg_e]
    print len(output)
    return output

def groupNeighborhoods(data):
    grouped = []
    for i in data['NAME']:
        if i == "Downtown":
            grouped.append("Downtown")
        elif i == "Fairview" or i == "Mount Pleasant":
            grouped.append("Fairview - Mount Pleasant")
        elif i == 'Kensington-Cedar Cottage' or i == "Riley Park":
            grouped.append("Kensington-Cedar Cottage - Riley Park")
        elif i == "Stanley Park":
            grouped.append("Stanley Park")
        elif i == "Sunset" or i == 'Victoria-Fraserview':
            grouped.append("Sunset - Victoria-Fraserview")
        elif i == "West End":
            grouped.append("West End")
        else:
            grouped.append(i)  
    

    print "length of group is: " +str(len(grouped))
    print "length of data is: " + str(len(data))

    data['hoodgrouped'] = grouped

    return data

def latitudeRank(data):
    def addcentroid(ifile):
        lon = [i.coords[0][0] for i in data.centroid] 
        lat = [i.coords[0][1] for i in data.centroid] 
        return lat,lon
    lat,lon = addcentroid(data)
    data['cntr_x'] = lon
    data['cntr_y'] = lat

    return data

# Zonal statistics for aggregating svf and building co2
def zonalstats(data):
    path2raster = os.getcwd()+'/diysco2-db/_main_/EPiCC-Data/'+'TRANSECT-RASTER-CO2e-BUILDINGS-1M-FILL.tif'
    ztype = 'bco2eyr'

    temp = data.copy()
    stats= zonal_stats(temp, path2raster, stats=['mean','min', 'max', 'count', 'median', 'majority', 'sum', 'range'])
    summarystats = [(f['mean'], f['min'], f['max'], f['count'], f['median'],f['majority'], f['sum'], f['range']) for f in stats]
    zonal_means = pd.DataFrame(summarystats)
    # ztype= "bco2e"
    zonal_means.columns = ['mean_'+ztype,'min_'+ztype, 'max_'+ztype,'cnt_'+ztype, 'med_'+ztype, 'maj_'+ztype, 'sum_'+ztype, 'rng_'+ztype]
    # zonal_means = zonal_means.fillna(-9999)
    zonal_means['rownum'] = [i for i in range(len(zonal_means))]
    temp['buid'] = [i for i in range(len(temp))]
    temp = GeoDataFrame.merge(temp, zonal_means, left_on="buid", right_on="rownum")
    print "calc bco2e"
    print "zonal statistics complete!"

    return temp
    
def adjustBco2(data, monthFactor, gridsize):
    gridsize2sqm = gridsize*gridsize
    gridScalingFactor = gridsize2sqm / 10000.0 # to get per hectare regardless of the grid size
    # monthFactor = 1.289849324

    data['bco2e_may'] = -9999
    for i in range(len(data['sum_bco2eyr'])):
        # if data['sum_bco2eyr'].iloc[i] != -9999:
        if pd.isnull(data['sum_bco2eyr'].iloc[i]) == False:
            # adjusting the buildings to may/june
            # data['bco2e_may'].iloc[i] = data['sum_bco2eyr'].iloc[i]/365/24 * 0.55
            # adjusting the buildings to may
            # data['bco2e_may'].iloc[i] = (((data['sum_bco2eyr'].iloc[i]/365.0)/24.0) * monthFactor) / gridScalingFactor
            data['bco2e_may'].iloc[i] = (((data['sum_bco2eyr'].iloc[i]/365.0)/24.0) * monthFactor) 
        else:
            # data['bco2e_may'].iloc[i] = -9999
            # data['bco2e_may'].iloc[i] = np.nan
            data['bco2e_may'].iloc[i] = 0


    return data

def calctotalco2emissions(data):
    data['bt_co2e'] = -9999
    for i in range(len(data)):
        # if data['bco2e_may'].iloc[i] != -9999:
        if pd.isnull(data['bco2e_may'].iloc[i]) == False:
            data['bt_co2e'].iloc[i] = data['bco2e_may'].iloc[i] +  data['e_10_14_hr'].iloc[i] # h10TO14_e
        else:
            # data['bt_co2e'].iloc[i] = -9999
            # data['bt_co2e'].iloc[i] = np.nan
            data['bt_co2e'].iloc[i] = 0

    return data

# calc absolute and relative difference:
def calcDiffs(data):
    # should I put in the total co2 here? 
    # data['absdiff'] = data['co2_avg_e'] - data['e_10_14_hr']
    # data['reldiff'] = (data['co2_avg_e'] - data['e_10_14_hr'] )/data['e_10_14_hr']
    data['absdiff'] = abs(data['co2_avg_e']) - abs(data['bt_co2e'])
    data['reldiff'] = (abs(data['co2_avg_e']) - abs(data['bt_co2e']) )/ abs(data['bt_co2e'])

    # calc the total differences***

    # make sure relative difference is adjusted
    for i in range(len(data)):
        if data['e_10_14_hr'].iloc[i] == 0:
             data['reldiff'].iloc[i] = 0

    return data


def writeProcessedData(data, experiment, gridsize):
    print "writing file out"
    data.to_file(os.getcwd()+'/diysco2-db/campaigns/'+experiment+'/diysco2-grid/' +'gridded_emissions_dev_'+str(gridsize)+'.shp')
    with open( os.getcwd()+'/diysco2-db/campaigns/'+experiment+'/diysco2-grid/' +'gridded_emissions_dev_'+str(gridsize)+'.geojson', 'w') as f:
        f.write(data.to_json())
    print("file written to:", os.getcwd()+'/diysco2-db/campaigns/'+experiment+'/diysco2-grid/' +'gridded_emissions_dev_'+str(gridsize)+'.geojson')


def generateEmissionsData(campaignJson, processAll=True, gridsize='Null'):
    gridlist = [{'gridsize':50, 'd':'grid_emissions_50m'},{'gridsize':100,'d':'grid_emissions_100m'},{'gridsize':200,'d':'grid_emissions_200m'},{'gridsize':400,'d':'grid_emissions_400m'}]
    eg = "Null"
    if gridsize == 50:
        eg = gridlist[0]
    elif gridsize == 100:
        eg = gridlist[1]
    elif gridsize == 200:
        eg = gridlist[2]
    elif gridsize == 400:
        eg = gridlist[3]

    campaignJson['insitu_calibration_start']= getCalibrationValues(campaignJson['data_path'],campaignJson['cal_start_initial'], campaignJson['cal_end_initial'], 'initial', campaignJson['experiment'])
    campaignJson['insitu_calibration_end']= getCalibrationValues(campaignJson['data_path'],campaignJson['cal_start_end'], campaignJson['cal_start_end'], 'end', campaignJson['experiment'])
    campaignJson['diysco2_filtered']= filterRaw(campaignJson['data_path'], campaignJson['start'], campaignJson['end'], campaignJson['experiment'])
    campaignJson['diysco2_filtered_points']= makePoints(campaignJson['experiment'])
    campaignJson['gridjoin_50m'] = makeGrid(campaignJson['diysco2_filtered_points'], campaignJson['experiment'], 50)
    campaignJson['gridjoin_100m'] = makeGrid(campaignJson['diysco2_filtered_points'], campaignJson['experiment'], 100)
    campaignJson['gridjoin_200m'] = makeGrid(campaignJson['diysco2_filtered_points'], campaignJson['experiment'], 200)
    campaignJson['gridjoin_400m'] = makeGrid(campaignJson['diysco2_filtered_points'], campaignJson['experiment'], 400)
    campaignJson['gridstats_50m'] = calcgeostats(campaignJson['gridjoin_50m'],campaignJson['experiment'], 50)
    campaignJson['gridstats_100m'] = calcgeostats(campaignJson['gridjoin_100m'],campaignJson['experiment'], 100)
    campaignJson['gridstats_200m'] = calcgeostats(campaignJson['gridjoin_200m'],campaignJson['experiment'], 200)
    campaignJson['gridstats_400m'] = calcgeostats(campaignJson['gridjoin_400m'],campaignJson['experiment'], 400)
    campaignJson['dt_traffic_hour_array'] = generateDtTrafficArray(campaignJson['traffic_hour_array'])
    campaignJson['e_traffic_hour_array'] = generateETrafficArray(campaignJson['traffic_hour_array'])

    if processAll == True:
        for i in gridlist:
            # calculate emissions
            print "calculating emissions"
            i_gridstats = 'gridstats_'+ str(i['gridsize'])+'m'
            campaignJson[i['d']] = roundVals( 
                    calculateTrafficEmissions(
                        mergeTrafficCountsToGrid( campaignJson[i_gridstats],
                            sumByGridId(getDistanceTravelled(
                                joinTrafficCounts(
                                    campaignJson[i_gridstats], campaignJson['utm10n']
                                    ), 
                                campaignJson['traffic_hour_array'], campaignJson['dt_traffic_hour_array']
                                ), 
                                campaignJson['traffic_hour_array'], campaignJson['dt_traffic_hour_array']
                            ), 
                    ), campaignJson['traffic_month_factor'],campaignJson['traffic_emissions_factor'],campaignJson['traffic_vehicle_efficiency'], campaignJson['dt_traffic_hour_array'], campaignJson['e_traffic_hour_array']
                ), campaignJson['dt_traffic_hour_array'], campaignJson['e_traffic_hour_array'], i['gridsize'] 
            )
            campaignJson[i['d']] = calcPeriodHourly(campaignJson[i['d']], 'e_10_14', 4)
            campaignJson[i['d']] = calcPeriodHourly(campaignJson[i['d']], 'e_1_24', 24)
            campaignJson[i['d']] = calcMeasuredEmissions(campaignJson[i['d']], campaignJson['densityOfAir'], campaignJson['molarDensityOfCo2'], campaignJson['molarMassOfAir'], campaignJson['backgroundCo2Concentrations'], campaignJson['rah'], i['gridsize'])
            campaignJson[i['d']] = addNeighborhoods(campaignJson[i['d']], campaignJson['utm10n'])
            campaignJson[i['d']] = groupNeighborhoods(campaignJson[i['d']])
            campaignJson[i['d']] = latitudeRank(campaignJson[i['d']])

            # building emissions
            campaignJson[i['d']] = adjustBco2(
                    zonalstats(
                        campaignJson[i['d']]
                    ), 
                    campaignJson['building_month_factor'], i['gridsize']
            )
            campaignJson[i['d']] = calctotalco2emissions(campaignJson[i['d']])
            campaignJson[i['d']] = calcDiffs(campaignJson[i['d']])
            writeProcessedData(campaignJson[i['d']], campaignJson['experiment'], i['gridsize'])
            print "finished"
    elif processAll == False:
        print "calculating emissions"
        i_gridstats = 'gridstats_'+ eg['gridsize']+'m'
        campaignJson[eg['d']] = roundVals( 
                calculateTrafficEmissions(
                    mergeTrafficCountsToGrid( campaignJson[i_gridstats],
                        sumByGridId(getDistanceTravelled(
                            joinTrafficCounts(
                                campaignJson[i_gridstats], campaignJson['utm10n']
                                ), 
                            campaignJson['traffic_hour_array'], campaignJson['dt_traffic_hour_array']
                            ), 
                            campaignJson['traffic_hour_array'], campaignJson['dt_traffic_hour_array']
                        ), 
                ), campaignJson['traffic_month_factor'],campaignJson['traffic_emissions_factor'],campaignJson['traffic_vehicle_efficiency'], campaignJson['dt_traffic_hour_array'], campaignJson['e_traffic_hour_array']
            ), campaignJson['dt_traffic_hour_array'], campaignJson['e_traffic_hour_array'], eg['gridsize'] 
        )
        campaignJson[eg['d']] = calcPeriodHourly(campaignJson[eg['d']], 'e_10_14', 4)
        campaignJson[eg['d']] = calcPeriodHourly(campaignJson[eg['d']], 'e_1_24', 24)
        campaignJson[eg['d']] = calcMeasuredEmissions(campaignJson[eg['d']], campaignJson['densityOfAir'], campaignJson['molarDensityOfCo2'], campaignJson['molarMassOfAir'], campaignJson['backgroundCo2Concentrations'], campaignJson['rah'], eg['gridsize'])
        campaignJson[eg['d']] = addNeighborhoods(campaignJson[eg['d']], campaignJson['utm10n'])
        campaignJson[eg['d']] = groupNeighborhoods(campaignJson[eg['d']])
        campaignJson[eg['d']] = latitudeRank(campaignJson[eg['d']])

        # building emissions
        campaignJson[eg['d']] = adjustBco2(
                zonalstats(
                    campaignJson[eg['d']]
                ), 
                campaignJson['building_month_factor'], eg['gridsize']
        )
        campaignJson[eg['d']] = calctotalco2emissions(campaignJson[eg['d']])
        campaignJson[eg['d']] = calcDiffs(campaignJson[eg['d']])
        writeProcessedData(campaignJson[eg['d']], campaignJson['experiment'], eg['gridsize'])

''' ----- json specs ---- '''

campaign_20150528 = {
    'experiment': '150528',
    'utm10n': {'init': 'epsg:26910', 'no_defs': True},
    'wgs84': {'datum':'WGS84', 'no_defs':True, 'proj':'longlat'},
    'data_path':  os.getcwd()+'/diysco2-db/campaigns/150528/diysco2-raw/',
    'cal_start_initial': '2015-05-28 21:19:00',
    'cal_end_initial': '2015-05-28 21:24:00',
    'cal_start_end': '2015-05-28 21:19:00',
    'cal_end_end': '2015-05-28 21:24:00',
    'start':'2015-05-28 17:33:00',
    'end': '2015-05-28 21:24:00',
    'insitu_calibration_start': 'Null',
    'insitu_calibration_end': 'Null',
    'diysco2_filtered': 'Null',
    'diysco2_filtered_points': 'Null',
    'gridjoin_50m': 'Null',
    'gridjoin_100m': 'Null',
    'gridjoin_200m': 'Null',
    'gridjoin_400m': 'Null',
    'gridstats_50m': 'Null',
    'gridstats_100m': 'Null',
    'gridstats_200m': 'Null',
    'gridstats_400m': 'Null',
    'traffic_hour_array': ['tc_1','tc_10','tc_10_14','tc_10_16','tc_11','tc_12','tc_13','tc_14','tc_15','tc_16','tc_17','tc_17_18','tc_18','tc_19','tc_1_24','tc_2','tc_20','tc_21','tc_22','tc_23','tc_24','tc_3','tc_4','tc_5','tc_6','tc_7','tc_8','tc_8_9','tc_9'],
    'dt_traffic_hour_array': 'Null',
    'e_traffic_hour_array': 'Null',
    'grid_emissions_50m': 'Null',
    'grid_emissions_100m': 'Null',
    'grid_emissions_200m': 'Null',
    'grid_emissions_400m': 'Null',
    'traffic_month_factor': 1.0216, # for transportation emissions in May:
    'traffic_emissions_factor': 2.175, # kg/l
    'traffic_vehicle_efficiency': 0.000129, # l / m
    'grid_scaling_factor': 10000.0, # to get per hectare regardless of grid size
    'densityOfAir':1.18767, # (kg m-3)
    'molarDensityOfCo2': 44.01, # g mol-1
    'molarMassOfAir': 28.9645, # g mol-1
    'backgroundCo2Concentrations': 399.45,
    'rah': 34.14,
    'building_month_factor': 0.6363
}

generateEmissionsData(campaign_20150528)


 



campaign_20160318 = {
    'experiment': '160318',
    'utm10n': {'init': 'epsg:26910', 'no_defs': True},
    'wgs84': {'datum':'WGS84', 'no_defs':True, 'proj':'longlat'},
    'data_path':  os.getcwd()+'/diysco2-db/campaigns/160318/diysco2-raw/',
    'cal_start_initial': '2016-03-18 17:24:00',
    'cal_end_initial': '2016-03-18 17:30:00',
    'cal_start_end': '2016-03-18 21:32:00',
    'cal_end_end': '2016-03-18 21:37:00',
    'start':'2016-03-18 17:23:00',
    'end': '2016-03-18 21:38:00',
    'insitu_calibration_start': 'Null',
    'insitu_calibration_end': 'Null',
    'diysco2_filtered': 'Null',
    'diysco2_filtered_points': 'Null',
    'gridjoin_50m': 'Null',
    'gridjoin_100m': 'Null',
    'gridjoin_200m': 'Null',
    'gridjoin_400m': 'Null',
    'gridstats_50m': 'Null',
    'gridstats_100m': 'Null',
    'gridstats_200m': 'Null',
    'gridstats_400m': 'Null',
    'traffic_hour_array': ['tc_1','tc_10','tc_10_14','tc_10_16','tc_11','tc_12','tc_13','tc_14','tc_15','tc_16','tc_17','tc_17_18','tc_18','tc_19','tc_1_24','tc_2','tc_20','tc_21','tc_22','tc_23','tc_24','tc_3','tc_4','tc_5','tc_6','tc_7','tc_8','tc_8_9','tc_9'],
    'dt_traffic_hour_array': 'Null',
    'e_traffic_hour_array': 'Null',
    'grid_emissions_50m': 'Null',
    'grid_emissions_100m': 'Null',
    'grid_emissions_200m': 'Null',
    'grid_emissions_400m': 'Null',
    'traffic_month_factor': 0.998509618, # for transportation emissions in May:
    'traffic_emissions_factor': 2.175, # kg/l
    'traffic_vehicle_efficiency': 0.000129, # l / m
    'grid_scaling_factor': 10000.0, # to get per hectare regardless of grid size
    'densityOfAir':1.18767, # (kg m-3) ,
    'molarDensityOfCo2': 44.01, # g mol-1,
    'molarMassOfAir': 28.9645, # g mol-1,
    'backgroundCo2Concentrations': 420.24, #ppm,
    'rah': 56.12, # s m-1,
    'building_month_factor': 1.289849324,
}
generateEmissionsData(campaign_20160318)




# campaign_20150528['insitu_calibration_start']= getCalibrationValues(campaign_20150528['data_path'],campaign_20150528['cal_start_initial'], campaign_20150528['cal_end_initial'], 'initial', campaign_20150528['experiment'])
# campaign_20150528['insitu_calibration_end']= getCalibrationValues(campaign_20150528['data_path'],campaign_20150528['cal_start_end'], campaign_20150528['cal_start_end'], 'end', campaign_20150528['experiment'])
# campaign_20150528['diysco2_filtered']= filterRaw(campaign_20150528['data_path'], campaign_20150528['start'], campaign_20150528['end'], campaign_20150528['experiment'])
# campaign_20150528['diysco2_filtered_points']= makePoints(campaign_20150528['data_path'], campaign_20150528['experiment'])
# campaign_20150528['gridjoin_50m'] = makeGrid(campaign_20150528['diysco2_filtered_points'], campaign_20150528['experiment'], 50)
# campaign_20150528['gridjoin_100m'] = makeGrid(campaign_20150528['diysco2_filtered_points'], campaign_20150528['experiment'], 100)
# campaign_20150528['gridjoin_200m'] = makeGrid(campaign_20150528['diysco2_filtered_points'], campaign_20150528['experiment'], 200)
# campaign_20150528['gridjoin_400m'] = makeGrid(campaign_20150528['diysco2_filtered_points'], campaign_20150528['experiment'], 400)
# campaign_20150528['gridstats_50m'] = calcgeostats(campaign_20150528['gridjoin_50m'],campaign_20150528['experiment'], 50)
# campaign_20150528['gridstats_100m'] = calcgeostats(campaign_20150528['gridjoin_100m'],campaign_20150528['experiment'], 100)
# campaign_20150528['gridstats_200m'] = calcgeostats(campaign_20150528['gridjoin_200m'],campaign_20150528['experiment'], 200)
# campaign_20150528['gridstats_400m'] = calcgeostats(campaign_20150528['gridjoin_400m'],campaign_20150528['experiment'], 400)
# campaign_20150528['dt_traffic_hour_array'] = generateDtTrafficArray(campaign_20150528['traffic_hour_array'])
# campaign_20150528['e_traffic_hour_array'] = generateETrafficArray(campaign_20150528['traffic_hour_array'])

# # calculate emissions
# campaign_20150528['grid_emissions_50m'] = roundVals( 
#         calculateTrafficEmissions(
#             mergeTrafficCountsToGrid( campaign_20150528['gridstats_50m'],
#                 sumByGridId(getDistanceTravelled(
#                     joinTrafficCounts(
#                         campaign_20150528['gridstats_50m'], campaign_20150528['utm10n']
#                         ), 
#                     campaign_20150528['traffic_hour_array'], campaign_20150528['dt_traffic_hour_array']
#                     ), 
#                     campaign_20150528['traffic_hour_array'], campaign_20150528['dt_traffic_hour_array']
#                 ), 
#         ), campaign_20150528['traffic_month_factor'],campaign_20150528['traffic_emissions_factor'],campaign_20150528['traffic_vehicle_efficiency'], campaign_20150528['dt_traffic_hour_array'], campaign_20150528['e_traffic_hour_array']
#     ), campaign_20150528['dt_traffic_hour_array'], campaign_20150528['e_traffic_hour_array'], 50 
# )
# campaign_20150528['grid_emissions_50m'] = calcPeriodHourly(campaign_20150528['grid_emissions_50m'], 'e_10_14', 4)
# campaign_20150528['grid_emissions_50m'] = calcPeriodHourly(campaign_20150528['grid_emissions_50m'], 'e_1_24', 24)
# campaign_20150528['grid_emissions_50m'] = calcMeasuredEmissions(campaign_20150528['grid_emissions_50m'], campaign_20150528['densityOfAir'], campaign_20150528['molarDensityOfCo2'], campaign_20150528['molarMassOfAir'], campaign_20150528['backgroundCo2Concentrations'], campaign_20150528['rah'], 50)
# campaign_20150528['grid_emissions_50m'] = addNeighborhoods(campaign_20150528['grid_emissions_50m'], campaign_20150528['utm10n'])
# campaign_20150528['grid_emissions_50m'] = groupNeighborhoods(campaign_20150528['grid_emissions_50m'])
# campaign_20150528['grid_emissions_50m'] = latitudeRank(campaign_20150528['grid_emissions_50m'])
# # building emissions
# campaign_20150528['grid_emissions_50m'] = adjustBco2(
#         zonalstats(
#             campaign_20150528['grid_emissions_50m']
#         ), 
#         campaign_20150528['building_month_factor'], 50
# )
# campaign_20150528['grid_emissions_50m'] = calctotalco2emissions(campaign_20150528['grid_emissions_50m'])
# campaign_20150528['grid_emissions_50m'] = calcDiffs(campaign_20150528['grid_emissions_50m'])
# writeProcessedData(campaign_20150528['grid_emissions_50m'], campaign_20150528['experiment'], 50)




# campaign_20150528['grid_emissions_100m'] = roundVals( 
#         calculateTrafficEmissions(
#             mergeTrafficCountsToGrid( campaign_20150528['gridstats_100m'],
#                 sumByGridId( getDistanceTravelled(    
#                     joinTrafficCounts(
#                         campaign_20150528['gridstats_100m'], campaign_20150528['utm10n']
#                         ), 
#                     campaign_20150528['traffic_hour_array'], campaign_20150528['dt_traffic_hour_array']
#                     ), 
#                     campaign_20150528['traffic_hour_array'], campaign_20150528['dt_traffic_hour_array']
#                 ), 
#         ), campaign_20150528['traffic_month_factor'],campaign_20150528['traffic_emissions_factor'],campaign_20150528['traffic_vehicle_efficiency'], campaign_20150528['dt_traffic_hour_array'], campaign_20150528['e_traffic_hour_array']
#     ), campaign_20150528['dt_traffic_hour_array'], campaign_20150528['e_traffic_hour_array'], 100
# )
# campaign_20150528['grid_emissions_200m'] = roundVals( 
#         calculateTrafficEmissions(
#             mergeTrafficCountsToGrid( campaign_20150528['gridstats_200m'],
#                 sumByGridId( getDistanceTravelled(
#                     joinTrafficCounts(
#                         campaign_20150528['gridstats_200m'], campaign_20150528['utm10n']
#                         ),
#                     campaign_20150528['traffic_hour_array'], campaign_20150528['dt_traffic_hour_array']
#                     ), 
#                     campaign_20150528['traffic_hour_array'], campaign_20150528['dt_traffic_hour_array']
#                 ), 
#         ), campaign_20150528['traffic_month_factor'],campaign_20150528['traffic_emissions_factor'],campaign_20150528['traffic_vehicle_efficiency'], campaign_20150528['dt_traffic_hour_array'], campaign_20150528['e_traffic_hour_array']
#     ), campaign_20150528['dt_traffic_hour_array'], campaign_20150528['e_traffic_hour_array'], 200
# )
# campaign_20150528['grid_emissions_400m'] = roundVals( 
#         calculateTrafficEmissions(
#             mergeTrafficCountsToGrid( campaign_20150528['gridstats_400m'],
#                 sumByGridId( getDistanceTravelled(
#                     joinTrafficCounts(
#                         campaign_20150528['gridstats_400m'], campaign_20150528['utm10n']
#                         ),
#                     campaign_20150528['traffic_hour_array'], campaign_20150528['dt_traffic_hour_array']
#                     ), 
#                     campaign_20150528['traffic_hour_array'], campaign_20150528['dt_traffic_hour_array']
#                 ), 
#         ), campaign_20150528['traffic_month_factor'],campaign_20150528['traffic_emissions_factor'],campaign_20150528['traffic_vehicle_efficiency'], campaign_20150528['dt_traffic_hour_array'], campaign_20150528['e_traffic_hour_array']
#     ), campaign_20150528['dt_traffic_hour_array'], campaign_20150528['e_traffic_hour_array'], 400
# )
# calculate distance traveled
 
