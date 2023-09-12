#!/bin/env lua

-- The original NewShip.mtl is somewhat hacked together. It uses some of the
-- PBR extensions, but not quite in the standard way. In particular, the Pr
-- (roughness) parameter includes a shininess value. This script converts the
-- Pr value back to roughness.
--
-- Run as 
-- $ lua mangle.lua > NewShip.mtl
--
-- For conversion functions, see
-- - "Crash Course in BRDF Implementation", Jakub Boksansky, 2021
--   https://boksajak.github.io/blog/BRDF
--
-- - "Physically Based Shading at Disney", Brent Burley, 2012
--   https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf

local shininess_to_roughness_ = function( aShininess )
	return math.pow( 2.0 / (aShininess + 2.0), 0.25 );
end

local fin = io.open( "NewShip.mtl.orig" );

local first = true;
for line in fin:lines() do

	local a, b = line:match( "([^ \t]+)[ \t]+(.*)" );

	if "Pr" == a then
		shininess = tonumber(b);
		roughness = shininess_to_roughness_( shininess );

		print( string.format( "Pr %f", roughness ) );
	elseif "#" == a then
		print( line );
		if first then
			print( "# MODIFIED/fix: converted Pr from shininess to roughness" );
			first = false;
		end
	else
		print( line );
	end
end

fin:close();

-- EOF
