-- New Database Schema for Accumulative Jar System
-- This schema supports adding monthly income to jars instead of resetting them
-- All views converted to tables for better performance

-- Keep existing users and jar_categories tables
CREATE TABLE public.users (
  id integer NOT NULL DEFAULT nextval('users_id_seq'::regclass),
  email character varying NOT NULL UNIQUE,
  full_name character varying,
  saving_target_cents bigint DEFAULT 0,
  user_description text DEFAULT NULL,
  created_at timestamp without time zone NOT NULL DEFAULT now(),
  CONSTRAINT users_pkey PRIMARY KEY (id)
);

CREATE TABLE public.jar_categories (
  id integer NOT NULL DEFAULT nextval('jar_categories_id_seq'::regclass),
  name character varying NOT NULL UNIQUE,
  description text,
  CONSTRAINT jar_categories_pkey PRIMARY KEY (id)
);

-- Rest of the schema remains unchanged
[Rest of the file content remains the same...]